"""Driver for the CoaXPress Grabber camera interface (ARTIQ 8 / legacy compiler).

Ported from the ARTIQ 10 (NAC3) ``artiq.coredevice.cxp_grabber`` driver. The
behaviour is identical; only the compiler-facing style was changed:

  * NAC3 ``@compile`` / ``KernelInvariant[T]`` / ``Kernel[T]`` -> legacy
    ``kernel_invariants`` set and plain ``@kernel`` methods (no type annotations).
  * NAC3 ``@extern`` syscalls -> legacy ``@syscall`` with ``T*`` type annotations.
    Flags follow the legacy convention (``rtio.py``/``i2c.py``): register I/O that
    does not touch kernel-visible memory but *can* raise ``CXPError`` uses
    ``{"nowrite"}`` (no ``nounwind``); the frame/XML download syscalls write into a
    caller-provided buffer, so they carry no flags.
  * ``self.core.delay(dt)`` -> free ``delay(dt)`` / ``delay_mu(...)``.

The real-time ROI-sum path (``setup_roi`` / ``gate_roi`` / ``input_mu``) mirrors the
legacy Camera-Link ``grabber.py`` exactly.
"""

from numpy import array, int32, int64

from artiq.language.core import *
from artiq.language.types import *
from artiq.coredevice.rtio import rtio_output, rtio_input_timestamped_data


class OutOfSyncException(Exception):
    """Raised when an incorrect number of ROI engine outputs has been
    retrieved from the RTIO input FIFO."""
    pass


class CXPGrabberTimeoutException(Exception):
    """Raised when a timeout occurs while attempting to read CoaXPress Grabber
    RTIO input events."""
    pass


# ---------------------------------------------------------------------------
# Syscalls (implemented in the Kasli-SoC firmware, libksupport kernel/cxp.rs).
# These do CoaXPress control-channel / ROI-viewer I/O and CAN raise CXPError, so
# they must NOT be marked "nounwind". Register read/write/arm do not modify
# kernel-visible memory -> "nowrite"; the download syscalls fill a caller buffer
# -> no flags.
# ---------------------------------------------------------------------------
@syscall
def cxp_download_xml_file(dest: TInt32, buffer: TList(TInt32)) -> TInt32:
    raise NotImplementedError("syscall not simulated")


@syscall(flags={"nowrite"})
def cxp_read32(dest: TInt32, addr: TInt32) -> TInt32:
    raise NotImplementedError("syscall not simulated")


@syscall(flags={"nowrite"})
def cxp_write32(dest: TInt32, addr: TInt32, val: TInt32) -> TNone:
    raise NotImplementedError("syscall not simulated")


@syscall(flags={"nowrite"})
def cxp_start_roi_viewer(dest: TInt32, x0: TInt32, y0: TInt32, x1: TInt32, y1: TInt32) -> TNone:
    raise NotImplementedError("syscall not simulated")


@syscall
def cxp_download_roi_viewer_frame(dest: TInt32, buffer: TList(TInt64)) -> TTuple([TInt32, TInt32, TInt32]):
    raise NotImplementedError("syscall not simulated")


@rpc
def write_file(data, file_path):
    """
    Write big-endian encoded data to PC

    :param data: a list of 32-bit integers
    :param file_path: a relative path on PC
    """
    array(data, dtype=">i").tofile(file_path)


@rpc
def write_pgm(frame, file_path, pixel_width):
    """
    Write the frame as a PGM file to PC (host helper).

    :param frame: a 2D numpy array of 32-bit integers
    :param file_path: a relative path on PC
    :param pixel_width: bit depth that the PGM will use (8 or 16)
    """
    from numpy import ndarray
    if not isinstance(frame, ndarray):
        raise ValueError("Frame must be a numpy array")

    if pixel_width == 8:
        frame = frame.astype("u1")
    elif pixel_width == 16:
        # PGM uses big-endian
        frame = frame.astype(">u2")
    else:
        raise ValueError("PGM file format only supports 8-bit or 16-bit per pixel")

    # Save as PGM binary variant
    # https://en.wikipedia.org/wiki/Netpbm#Description
    with open(file_path, "wb") as file:
        max_value = (2**pixel_width) - 1
        width, height = len(frame[0]), len(frame)
        file.write("P5\n{} {}\n{}\n".format(width, height, max_value).encode("ASCII"))
        file.write(frame.tobytes())


class CXPGrabber:
    """Driver for the CoaXPress Grabber camera interface."""
    kernel_invariants = {
        "core", "channel", "trigger_ch", "roi_config_ch", "roi_gating_ch", "sentinel"
    }

    def __init__(self, dmgr, channel, core_device="core", count_width=31):
        self.core = dmgr.get(core_device)

        self.channel = channel
        self.trigger_ch = channel
        self.roi_config_ch = channel + 1
        self.roi_gating_ch = channel + 2

        # This value is inserted by the gateware to mark the start of a series of
        # ROI engine outputs for one video frame.
        self.sentinel = int32(int64(2**count_width))

    @staticmethod
    def get_rtio_channels(channel, **kwargs):
        return [
            (channel, "Trigger"),
            (channel + 1, "ROI coordinates"),
            (channel + 2, "ROI mask"),
        ]

    @kernel
    def send_cxp_linktrigger(self, linktrigger, extra_linktrigger=False):
        """
        Send a CoaXPress fixed-latency linktrigger to the camera.

        :param linktrigger: linktrigger type. 0-1 when extra_linktrigger is False;
                            0-3 when extra_linktrigger is True. In CXP v1.x,
                            linktrigger0 was "rising edge" and linktrigger1 "falling
                            edge".
        :param extra_linktrigger: set to True when ExtraLsTriggerEnable is 1 on the
                            camera.
        """
        extra_linktrigger_mask = 1 if extra_linktrigger else 0
        rtio_output(self.trigger_ch << 8, linktrigger << 2 | extra_linktrigger_mask)

    @kernel
    def setup_roi(self, n, x0, y0, x1, y1):
        """
        Defines the coordinates of a ROI.

        The coordinates are set around the current position of the RTIO time
        cursor.

        The user must keep the ROI engine disabled for a duration of more
        than one video frame after calling this function, as the output
        generated for that video frame is undefined.

        Advances the timeline by 4 coarse RTIO cycles.
        """
        c = int64(self.core.ref_multiplier)
        rtio_output(self.roi_config_ch << 8 | (4 * n + 0), x0)
        delay_mu(c)
        rtio_output(self.roi_config_ch << 8 | (4 * n + 1), y0)
        delay_mu(c)
        rtio_output(self.roi_config_ch << 8 | (4 * n + 2), x1)
        delay_mu(c)
        rtio_output(self.roi_config_ch << 8 | (4 * n + 3), y1)
        delay_mu(c)

    @kernel
    def gate_roi(self, mask):
        """
        Defines which ROI engines produce input events.

        At the end of each video frame, the output from each ROI engine that
        has been enabled by the mask is enqueued into the RTIO input FIFO.

        This function sets the mask at the current position of the RTIO time
        cursor.

        Setting the mask using this function is atomic; in other words,
        if the system is in the middle of processing a frame and the mask
        is changed, the processing will complete using the value of the mask
        that it started with.

        :param mask: bitmask enabling or disabling each ROI engine.
        """
        rtio_output(self.roi_gating_ch << 8, mask)

    @kernel
    def gate_roi_pulse(self, mask, dt):
        """
        Sets a temporary mask for the specified duration (in seconds), then
        disables all ROI engines.
        """
        self.gate_roi(mask)
        delay(dt)
        self.gate_roi(0)

    @kernel
    def input_mu(self, data, timeout_mu=-1):
        """
        Retrieves the accumulated values for one frame from the ROI engines.
        Blocks until values are available or timeout is reached.

        The input list must be a list of integers of the same length as there
        are enabled ROI engines. This method replaces the elements of the
        input list with the outputs of the enabled ROI engines, sorted by
        number.

        If the number of elements in the list does not match the number of
        ROI engines that produced output, an exception will be raised during
        this call or the next.

        If the timeout is reached before data is available, the exception
        :exc:`CXPGrabberTimeoutException` is raised.

        :param timeout_mu: Timestamp at which a timeout will occur. Set to -1
                           (default) to disable timeout.
        """
        timestamp, sentinel = rtio_input_timestamped_data(timeout_mu, self.roi_gating_ch)
        if timestamp == -1:
            raise CXPGrabberTimeoutException(
                "Timeout before CoaXPress Grabber frame available")
        if sentinel != self.sentinel:
            raise OutOfSyncException

        for i in range(len(data)):
            timestamp, roi_output = rtio_input_timestamped_data(timeout_mu, self.roi_gating_ch)
            if roi_output == self.sentinel:
                raise OutOfSyncException
            if timestamp == -1:
                raise CXPGrabberTimeoutException(
                    "Timeout retrieving ROIs (attempting to read more ROIs than enabled?)")
            data[i] = roi_output

    @kernel
    def read32(self, address):
        """
        Read a 32-bit value from a camera register.

        .. warning:: This is NOT a real-time operation.

        :param address: 32-bit register address to read from
        :returns: 32-bit value from register
        """
        return cxp_read32(self.channel >> 16, address)

    @kernel
    def write32(self, address, value):
        """
        Write a 32-bit value to a camera register.

        .. warning:: This is NOT a real-time operation.

        :param address: 32-bit register address to write to
        :param value: 32-bit value to be written
        """
        cxp_write32(self.channel >> 16, address, value)

    @kernel
    def read_local_xml(self, buffer):
        """
        Read the XML setting file from the camera if available.
        Data will be in 32-bit big-endian encoding. The file format may be .zip
        or .xml depending on the camera model.

        .. warning:: This is NOT a real-time operation.

        :param buffer: list to be filled
        :returns: number of 32-bit words read
        """
        return cxp_download_xml_file(self.channel >> 16, buffer)

    @kernel
    def start_roi_viewer(self, x0, y0, x1, y1):
        """
        Defines the coordinates of the ROI viewer and starts the capture.

        Unlike :meth:`setup_roi`, the ROI viewer has a maximum size of 4096 pixels.

        .. warning:: This is NOT a real-time operation.
        """
        cxp_start_roi_viewer(self.channel >> 16, x0, y0, x1, y1)

    @kernel
    def read_roi_viewer_frame(self, frame):
        """
        Read the ROI viewer frame.

        The user must :meth:`start_roi_viewer` and trigger the camera before the
        frame is available.

        .. warning:: This is NOT a real-time operation.

        :param frame: a 2D array of 32-bit integers
        :returns: the frame bit depth
        """
        buffer = [int64(0)] * 1024
        width, height, pixel_width = cxp_download_roi_viewer_frame(
            self.channel >> 16, buffer)
        if height != len(frame) or width != len(frame[0]):
            raise ValueError(
                "The frame matrix size is not the same as ROI viewer frame size")

        for y in range(height):
            for x in range(width):
                pos = x + (y * width)
                # each buffer element holds 4 pixels
                frame[y][x] = int32((buffer[pos // 4] >> (16 * (pos % 4))) & int64(0xFFFF))
        return pixel_width
