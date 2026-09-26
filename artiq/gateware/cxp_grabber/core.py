from migen import *
from migen.genlib.coding import PriorityEncoder
from migen.genlib.cdc import MultiReg, PulseSynchronizer
from misoc.interconnect.csr import *
from misoc.interconnect.stream import AsyncFIFO, StrideConverter, Endpoint
from misoc.cores.coaxpress.common import word_width
from misoc.cores.coaxpress.core import HostTXCore, HostRXCore

from math import log, ceil
from operator import add


def cfg_layout(x_size, y_size):
    return [
        ("x0", x_size, DIR_M_TO_S),
        ("x1", x_size, DIR_M_TO_S),
        ("y0", y_size, DIR_M_TO_S),
        ("y1", y_size, DIR_M_TO_S),
    ]


class CXPHostCore(Module, AutoCSR):
    def __init__(self, tx_phy, rx_phy, clk_freq, command_buffer_depth=32, nrxslot=4):
        # control buffer is only 32 words (128 bytes) wide for compatibility with CXP 1.x compliant devices
        # Section 12.1.6 (CXP-001-2021)
        self.buffer_depth, self.nslots = command_buffer_depth, nrxslot

        self.submodules.tx = HostTXCore(tx_phy, command_buffer_depth, clk_freq, False)
        self.submodules.rx = HostRXCore(rx_phy, command_buffer_depth, nrxslot, False)

    def get_tx_port(self):
        return self.tx.writer.mem.get_port(write_capable=True)

    def get_rx_port(self):
        return self.rx.command_reader.mem.get_port(write_capable=False)

    def get_mem_size(self):
        return word_width * self.buffer_depth * self.nslots // 8


class MonoPixelTracker(Module):
    def __init__(self, pixel_source, res_width):

        max_pixel_width = len(pixel_source.px0)
        n_pixels = len(pixel_source.px_stb)
        self.tracked_pixels = [
            Record(
                [
                    ("x", res_width),
                    ("y", res_width),
                    ("gray", max_pixel_width),
                    ("stb", 1),
                    ("eof", 1),
                ]
            )
            for _ in range(n_pixels)
        ]

        # # #

        eof_r = Signal()
        y = Signal.like(self.tracked_pixels[0].y)
        self.sync += [
            eof_r.eq(pixel_source.eof),
            # The n pixels in pixel_source are on the same y level, they can share the same y
            If(pixel_source.eof,
                y.eq(y.reset),
            ).Elif(pixel_source.eol,
                y.eq(y + 1),
            ),
        ]

        for i, px in enumerate(self.tracked_pixels):
            # Offset init x position
            x = Signal.like(self.tracked_pixels[0].x)
            x.reset = i

            self.comb += px.eof.eq(eof_r)
            self.sync += [
                px.gray.eq(getattr(pixel_source, f"px{i}")),
                px.stb.eq(pixel_source.px_stb[i]),
                px.x.eq(x),
                px.y.eq(y),
                If(pixel_source.eof | pixel_source.eol,
                    x.eq(x.reset),
                ).Elif(pixel_source.px_stb[i],
                    x.eq(x + n_pixels),
                ),
            ]


class ROICropper(Module):
    def __init__(self, tracked_pixels):
        self.cfg = Record(
            cfg_layout(len(tracked_pixels[0].x), len(tracked_pixels[0].y))
        )

        # # #

        # Pixel cropping
        self.roi_pixels = [
            Record(
                [
                    ("x_good", 1),
                    ("y_good", 1),
                    ("gray", len(tracked_pixels[0].gray)),
                    ("stb", 1),
                    ("eof", 1),
                ]
            )
            for _ in range(len(tracked_pixels))
        ]

        for i, (roi, px) in enumerate(zip(self.roi_pixels, tracked_pixels)):
            self.sync += [
                roi.x_good.eq(0),
                If((self.cfg.x0 <= px.x) & (px.x < self.cfg.x1), roi.x_good.eq(1)),
                # all tracked pixels are on the same line, only set y_good when changing lines
                If((self.cfg.y1 == px.y) | roi.eof,
                    roi.y_good.eq(0),
                ).Elif((self.cfg.y0 == px.y) & px.stb,
                    # To prevent y_good high when waiting for next frame (e.g. setting y0 == 0)
                    # only set y_good when px.stb is high
                    roi.y_good.eq(1),
                ),
                roi.gray.eq(px.gray),
                roi.stb.eq(px.stb),
                roi.eof.eq(px.eof),
            ]


class PipelinedAdder(Module):
    def __init__(self, regs, operand_per_stage):
        self.n_stage = ceil(log(len(regs), operand_per_stage))

        # deep copy to prevent deleting signals
        operands = regs.copy()
        accu = []
        for _ in range(self.n_stage):
            while len(operands) > 0:
                buf = []
                for _ in range(operand_per_stage):
                    if len(operands) > 0:
                        buf.append(operands.pop(0))

                sum = Signal(value_bits_sign(reduce(add, buf)))
                self.sync += sum.eq(reduce(add, buf))
                accu.append(sum)

            # prepare for next stage
            operands = accu.copy()
            accu = []

        assert len(operands) == 1
        self.o = operands[0]


class ROI(Module):
    """
    ROI Engine that accept multiple pixels each cycle. For each frame, accumulates pixels values within a
    rectangular region of interest, and reports the total.
    """

    def __init__(self, tracked_pixels, count_width):
        self.cfg = Record(
            cfg_layout(len(tracked_pixels[0].x), len(tracked_pixels[0].y))
        )

        self.out = Record(
            [
                ("update", 1),
                # registered output - can be used as CDC input
                ("count", count_width),
            ]
        )

        # # #

        self.submodules.cropper = cropper = ROICropper(tracked_pixels)
        self.sync += self.cfg.connect(cropper.cfg)

        counts = [Signal(count_width) for _ in range(len(cropper.roi_pixels))]

        for cnt, roi in zip(counts, cropper.roi_pixels):
            eof_r = Signal()
            self.sync += [
                eof_r.eq(roi.eof),
                If(eof_r,
                    cnt.eq(cnt.reset),
                ).Elif((roi.stb & roi.x_good & roi.y_good),
                    cnt.eq(cnt + roi.gray),
                ),
            ]

        self.submodules.adder = adder = PipelinedAdder(counts, 4)
        # Match adder and cnt delay
        eofs = [Signal() for _ in range(adder.n_stage + 1)]

        self.sync += eofs[0].eq(cropper.roi_pixels[0].eof)
        for last, curr in zip(eofs, eofs[1:]):
            self.sync += curr.eq(last)

        self.sync += [
            self.out.update.eq(eofs[-1]),
            If(eofs[-1], self.out.count.eq(adder.o)),
        ]


class ROIPacker(Module):
    """Efficiently pack pixel to fill the register before passing to downstream"""

    def __init__(self, tracked_pixels):
        n_pixels = len(tracked_pixels)
        max_pixel_width = len(tracked_pixels[0].gray)
        layout = [("data", n_pixels * max_pixel_width)]

        self.cfg = Record(
            cfg_layout(len(tracked_pixels[0].x), len(tracked_pixels[0].y))
        )

        self.source = Endpoint(layout)
        self.eof = Signal()

        # # #

        self.submodules.cropper = cropper = ROICropper(tracked_pixels)
        self.comb += self.cfg.connect(cropper.cfg)

        # Flatten ROI pixels to prepare packing pixel in ROI from LSB
        flatten_layout = [("data", max_pixel_width * n_pixels), ("eof", 1)]
        roi_px_flatten = Record(flatten_layout + [("stbs", n_pixels)])

        self.sync += [
            roi_px_flatten.data.eq(Cat([roi.gray for roi in cropper.roi_pixels])),
            roi_px_flatten.stbs.eq(
                Cat([(roi.stb & roi.x_good & roi.y_good) for roi in cropper.roi_pixels])
            ),
            roi_px_flatten.eof.eq(cropper.roi_pixels[0].eof),
        ]

        self.submodules.encoder = priority_encoder = PriorityEncoder(n_pixels)
        self.sync += priority_encoder.i.eq(
            Cat([(roi.stb & roi.x_good & roi.y_good) for roi in cropper.roi_pixels])
        )

        packed_px = Record(flatten_layout + [("stb", 1), ("sel", bits_for(n_pixels))])
        shift_px_cases = {
            i: packed_px.data.eq(roi_px_flatten.data[i * max_pixel_width :])
            for i in range(n_pixels)
        }
        self.sync += [
            packed_px.sel.eq(reduce(add, roi_px_flatten.stbs)),
            packed_px.eof.eq(roi_px_flatten.eof),
            packed_px.stb.eq(roi_px_flatten.stbs != 0),
            Case(priority_encoder.o, shift_px_cases),
        ]

        # Buffer loading

        n_buffers = 2
        max_cnts = n_buffers * n_pixels
        px_buffers = [Signal.like(self.source.data) for _ in range(n_buffers)]
        buffer_full = Signal()

        double_buffer_cases = {}
        for curr_cnt in range(max_cnts):
            buffer_loading_cases = {}
            target_buffer, px_in_buffer = divmod(curr_cnt, n_pixels)
            space_left = n_pixels - px_in_buffer

            # n_pixels >= packed_px.sel >= 1
            for px_to_load in range(1, n_pixels + 1):
                if px_to_load > space_left:
                    target = space_left
                else:
                    target = px_to_load

                data = packed_px.data[: target * max_pixel_width]
                buffer_loading_cases[px_to_load] = [
                    px_buffers[target_buffer][px_in_buffer * max_pixel_width :].eq(data)
                ]
                # It's not possible to full the next buffer full, no need to have two buffer full signal
                if px_to_load >= space_left:
                    buffer_loading_cases[px_to_load].append(buffer_full.eq(1))

                if px_to_load > space_left:
                    data = packed_px.data[
                        target * max_pixel_width : px_to_load * max_pixel_width
                    ]
                    buffer_loading_cases[px_to_load].append(
                        px_buffers[(target_buffer + 1) % n_buffers].eq(data)
                    )

            double_buffer_cases[curr_cnt] = Case(packed_px.sel, buffer_loading_cases)

        cnts = Signal(max=max(2, max_cnts))
        next_cnts = Signal.like(cnts)
        wcnts_reset = Signal()
        self.comb += next_cnts.eq(cnts + packed_px.sel)
        self.sync += [
            If(wcnts_reset,
                cnts.eq(cnts.reset),
            ).Elif(packed_px.stb,
                # cnts = (cnts + packed_px.sel) % max_cnts
                cnts.eq(next_cnts[: log2_int(max_cnts)]),
            ),
        ]
        self.sync += [
            buffer_full.eq(0),
            If(packed_px.stb, Case(cnts, double_buffer_cases)),
        ]

        # Select which buffer to output
        buffer_sel_cases = {
            i: self.source.data.eq(px_buffers[i]) for i in range(n_buffers)
        }
        buffer_sel = Signal(max=n_buffers)
        buffer_sel_reset = Signal()
        buffer_switch = Signal()
        self.comb += buffer_switch.eq(buffer_full & buffer_sel == n_buffers)
        self.sync += [
            If(buffer_switch | buffer_sel_reset,
                buffer_sel.eq(buffer_sel.reset),
            ).Elif(buffer_full,
                buffer_sel.eq(buffer_sel + 1),
            )
        ]
        self.comb += (Case(buffer_sel, buffer_sel_cases),)

        # Handle stb signals

        leftover_in_buffer = Signal()
        stb_leftover = Signal()
        self.comb += [
            # cnts % n_pixels
            leftover_in_buffer.eq(cnts[: log2_int(n_pixels)] != 0),
            self.source.stb.eq(buffer_full | stb_leftover),
        ]
        eof_r = Signal()
        self.sync += [
            stb_leftover.eq(packed_px.eof & leftover_in_buffer),
            # delay one cycle before reset make sure last pixel is stored in buffer
            eof_r.eq(packed_px.eof),
            wcnts_reset.eq(eof_r),
            buffer_sel_reset.eq(eof_r),
            self.eof.eq(eof_r),
        ]


class ROIViewer(Module, AutoCSR):
    def __init__(self, tracked_pixels, fifo_size=0x10000):
        self.arm = CSR()
        self.ready = CSR()
        self.x0 = CSRStorage(len(tracked_pixels[0].x))
        self.x1 = CSRStorage(len(tracked_pixels[0].x))
        self.y0 = CSRStorage(len(tracked_pixels[0].y))
        self.y1 = CSRStorage(len(tracked_pixels[0].y))

        max_pixel_width = len(tracked_pixels[0].gray)

        self.fifo_ack = CSR()
        self.fifo_data = CSRStatus(4 * max_pixel_width)
        self.fifo_stb = CSRStatus()

        # # #

        cdr = ClockDomainsRenamer("cxp_gt_rx")

        self.submodules.packer = packer = cdr(ROIPacker(tracked_pixels))
        cfg = Record(cfg_layout(len(tracked_pixels[0].x), len(tracked_pixels[0].y)))
        self.sync.cxp_gt_rx += cfg.connect(packer.cfg)
        self.specials += [
            MultiReg(self.x0.storage, cfg.x0, "cxp_gt_rx"),
            MultiReg(self.x1.storage, cfg.x1, "cxp_gt_rx"),
            MultiReg(self.y0.storage, cfg.y0, "cxp_gt_rx"),
            MultiReg(self.y1.storage, cfg.y1, "cxp_gt_rx"),
        ]

        layout = [("data", len(tracked_pixels) * max_pixel_width)]
        fifo_depth = ceil((fifo_size / layout_len(layout)))
        self.submodules.fifo = fifo = ClockDomainsRenamer(
            {"write": "cxp_gt_rx", "read": "sys"}
        )(AsyncFIFO(layout, fifo_depth))

        # Pixel output via CSR

        self.submodules.converter = converter = StrideConverter(
            layout, [("data", len(self.fifo_data.status))]
        )
        self.sync += [
            converter.source.ack.eq(self.fifo_ack.re),
            self.fifo_data.status.eq(converter.source.data),
            self.fifo_stb.status.eq(converter.source.stb),
        ]

        # Pipeline control
        pipeline = [fifo, converter]
        for s, d in zip(pipeline, pipeline[1:]):
            self.comb += s.source.connect(d.sink)

        connect_fifo = Signal()
        self.sync.cxp_gt_rx += [
            pipeline[0].sink.data.eq(packer.source.data),
            # Prevent loading FIFO when it's not needed
            If(connect_fifo,
                pipeline[0].sink.stb.eq(packer.source.stb),
            ).Else(
                pipeline[0].sink.stb.eq(0),
            ),
        ]

        self.submodules.arm_ps = arm_ps = PulseSynchronizer("sys", "cxp_gt_rx")
        self.submodules.ready_ps = ready_ps = PulseSynchronizer("cxp_gt_rx", "sys")
        self.sync += [
            arm_ps.i.eq(self.arm.re),
            If(ready_ps.o,
                self.ready.w.eq(1),
            ).Elif(self.ready.re,
                self.ready.w.eq(0),
            ),
        ]
        self.sync.cxp_gt_rx += [
            If(arm_ps.o,
                connect_fifo.eq(1),
            ).Elif(packer.eof,
                connect_fifo.eq(0),
            ),
            ready_ps.i.eq(packer.eof),
        ]
