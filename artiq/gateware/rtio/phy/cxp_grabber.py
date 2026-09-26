from migen import *
from migen.genlib.cdc import MultiReg 
from misoc.interconnect.csr import *
from misoc.cores.coaxpress.phy.asymmetric_gtx import HostTRXPHYs
from misoc.cores.coaxpress.phy.high_speed_gtx import HostRXPHYs
from misoc.cores.coaxpress.phy.low_speed_serdes import HostTXPHYs
from misoc.cores.coaxpress.core.stream_decoder import MultiChannelStreamDecoder

from artiq.gateware.rtio import rtlink
from artiq.gateware.rtio.phy.grabber import Serializer, Synchronizer
from artiq.gateware.cxp_grabber.core import ROI, ROIViewer, MonoPixelTracker

from math import ceil


class CXPGrabber(Module, AutoCSR):
    def __init__(
        self,
        refclk,
        gt_pads,
        sys_clk_freq,
        roi_engine_count=8,
        res_width=16,
        count_width=31,
        stream_fifo_size=0x10000,
        non_gt_tx_pads=None,
    ):
        assert count_width <= 31

        # Trigger rtio
        self.trigger = rtlink.Interface(rtlink.OInterface(3))
        self.trig_linktrig_mode = Signal(2)
        self.trig_extra_linktrig_en = Signal()
        self.trig_stb = Signal()

        # ROI rtio
         
        # 4 configs (x0, y0, x1, y1) per roi_engine
        self.config = rtlink.Interface(rtlink.OInterface(res_width, bits_for(4*roi_engine_count-1)))

        # select which roi engine can output rtio_input signal
        self.gate_data = rtlink.Interface(
            rtlink.OInterface(roi_engine_count),
            # the extra MSB bits is for sentinel
            rtlink.IInterface(count_width + 1, timestamped=False),
        )

        # # #
        self.sync.rio += [
            If(self.trigger.o.stb,
                Cat(self.trig_extra_linktrig_en, self.trig_linktrig_mode).eq(self.trigger.o.data)
            ),
            self.trig_stb.eq(self.trigger.o.stb),
        ]
         
        if non_gt_tx_pads:
            self.submodules.phy_tx = HostTXPHYs(non_gt_tx_pads, sys_clk_freq)
            self.submodules.phy_rx = HostRXPHYs(refclk, gt_pads, sys_clk_freq)
        else:
            self.submodules.phy = HostTRXPHYs(refclk, gt_pads, sys_clk_freq)

        # The host cores should be declared and connected externally to use csr group
        self.submodules.stream_decoder = stream_decoder = MultiChannelStreamDecoder(len(gt_pads), stream_fifo_size)

        # ROI Viewer
        cdr = ClockDomainsRenamer("cxp_gt_rx")
        self.submodules.pixel_tracker = pixel_tracker = cdr(MonoPixelTracker(stream_decoder.pixel_source, res_width))
        self.submodules.roi_viewer = ROIViewer(pixel_tracker.tracked_pixels)

        # ROI engines configuration and count gating

        # Share some trackers output to save resource while maintaining timing requirement
        roi_per_tracker_ratio = 4
        pixel_trackers = [
            cdr(MonoPixelTracker(stream_decoder.pixel_source, res_width))            
            for _ in range(ceil(roi_engine_count / roi_per_tracker_ratio))
        ]
        roi_engines = [
            cdr(ROI(pixel_trackers[i // roi_per_tracker_ratio].tracked_pixels, count_width))
            for i in range(roi_engine_count)
        ]
        self.submodules += roi_engines, pixel_trackers

        for n, roi in enumerate(roi_engines):
            cfg = roi.cfg
            for offset, target in enumerate([cfg.x0, cfg.y0, cfg.x1, cfg.y1]):
                roi_boundary = Signal.like(target)
                self.sync.rio += If(self.config.o.stb & (self.config.o.address == 4*n+offset),
                                roi_boundary.eq(self.config.o.data))
                self.specials += MultiReg(roi_boundary, target, "cxp_gt_rx")

        self.submodules.synchronizer = synchronizer = ClockDomainsRenamer({"cl" : "cxp_gt_rx"})(Synchronizer(roi_engines))
        self.submodules.serializer = serializer = Serializer(synchronizer.update, synchronizer.counts, self.gate_data.i)
        
        self.sync.rio += If(self.gate_data.o.stb, serializer.gate.eq(self.gate_data.o.data))
