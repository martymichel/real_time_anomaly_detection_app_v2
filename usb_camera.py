"""
USB Camera wrapper compatible with the IDS Camera API.

Provides a similar interface to camera.py for USB webcams using OpenCV.
"""

import cv2
import time
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class USBCamera:
    """USB Camera wrapper with API similar to IDS Peak Camera."""

    # Configuration from test script
    INDEX = 0
    BACKEND = cv2.CAP_MSMF  # Windows Media Foundation (best for USB on Windows)
    RES_LIST = [(2592, 1944), (2048, 1536), (1920, 1080), (1280, 720)]
    FPS_LIST = [60, 30]

    def __init__(
        self,
        camera_index: int = 0,
        backend: int = cv2.CAP_MSMF,
        preferred_resolution: Optional[Tuple[int, int]] = None,
        preferred_fps: Optional[int] = None
    ):
        """
        Initialize USB camera.

        Args:
            camera_index: Camera index (default: 0)
            backend: OpenCV backend (default: CAP_MSMF for Windows)
            preferred_resolution: Preferred (width, height) or None for auto-detect
            preferred_fps: Preferred FPS or None for auto-detect
        """
        self.camera_index = camera_index
        self.backend = backend
        self._cap: Optional[cv2.VideoCapture] = None
        self._acquisition_running = False
        self._width = 0
        self._height = 0
        self._fps = 0

        # Model name for compatibility with IDS API
        self._model_name = "USB Camera"

        # Open and configure camera
        self._open_camera(preferred_resolution, preferred_fps)

    def _open_camera(
        self,
        preferred_resolution: Optional[Tuple[int, int]] = None,
        preferred_fps: Optional[int] = None
    ):
        """Open camera and find best resolution/FPS combination."""
        self._cap = cv2.VideoCapture(self.camera_index, self.backend)

        if not self._cap.isOpened():
            raise RuntimeError(f"USB Camera {self.camera_index} konnte nicht geöffnet werden (Backend: {self.backend})")

        # Set MJPG for high resolution/FPS
        try:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except:
            logger.warning("MJPG FourCC konnte nicht gesetzt werden")

        # Note: Don't set BUFFERSIZE to 1 - causes MSMF errors
        # MSMF needs multiple buffers for stable operation

        # Try to find best resolution/FPS combination
        if preferred_resolution and preferred_fps:
            # User specified exact settings
            res_list = [preferred_resolution]
            fps_list = [preferred_fps]
        elif preferred_resolution:
            # User specified resolution, try all FPS
            res_list = [preferred_resolution]
            fps_list = self.FPS_LIST
        elif preferred_fps:
            # User specified FPS, try all resolutions
            res_list = self.RES_LIST
            fps_list = [preferred_fps]
        else:
            # Try all combinations
            res_list = self.RES_LIST
            fps_list = self.FPS_LIST

        best_config = None
        for w, h in res_list:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

            for fps in fps_list:
                self._cap.set(cv2.CAP_PROP_FPS, fps)
                time.sleep(0.05)

                # Test frames
                for _ in range(3):
                    self._cap.grab()

                ok, frame = self._cap.read()
                if ok and frame is not None:
                    actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

                    # Prefer exact match or highest resolution
                    if best_config is None or (actual_w * actual_h) > (best_config[0] * best_config[1]):
                        best_config = (actual_w, actual_h, actual_fps)
                        logger.info(f"Found valid config: {actual_w}x{actual_h} @{actual_fps:.1f} FPS")

        if best_config is None:
            raise RuntimeError("Keine gültige Kamera-Konfiguration gefunden")

        self._width, self._height, self._fps = best_config
        logger.info(f"USB Camera initialized: {self._width}x{self._height} @{self._fps:.1f} FPS (MSMF/MJPG)")

    def start_acquisition(self):
        """Start acquisition (for compatibility, USB cameras are always capturing)."""
        if self._acquisition_running:
            return

        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera not opened")

        self._acquisition_running = True
        logger.info("USB Camera acquisition started")

    def stop_acquisition(self):
        """Stop acquisition (for compatibility)."""
        if not self._acquisition_running:
            return

        self._acquisition_running = False
        logger.info("USB Camera acquisition stopped")

    def wait_for_image_view(self, timeout_ms: int = 5000) -> Tuple['USBImageView', 'USBBuffer']:
        """
        Wait for next frame (compatible with IDS Peak API).

        Returns:
            Tuple of (USBImageView, USBBuffer)
        """
        if not self._acquisition_running:
            raise RuntimeError("Acquisition not running")

        if self._cap is None:
            raise RuntimeError("Camera not opened")

        # Use read() instead of grab()/retrieve() - more reliable with MSMF
        start_time = time.time()
        max_retries = 5
        retry_count = 0

        while True:
            ok, frame = self._cap.read()

            if ok and frame is not None:
                # Wrap in compatible objects
                image_view = USBImageView(frame)
                buffer = USBBuffer(frame)
                return image_view, buffer

            # Increment retry counter
            retry_count += 1
            if retry_count >= max_retries:
                # Check timeout
                if (time.time() - start_time) * 1000 > timeout_ms:
                    raise TimeoutError(f"USB Camera timeout after {timeout_ms}ms (failed {max_retries} times)")
                # Reset retry counter
                retry_count = 0

            time.sleep(0.005)  # 5ms between retries

    def queue_buffer(self, buffer):
        """Queue buffer back (no-op for USB cameras, for compatibility)."""
        pass

    def kill_datastream_wait(self):
        """Kill waiting acquisition (no-op for USB cameras)."""
        pass

    def close(self):
        """Close camera and release resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("USB Camera closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if self._cap is not None:
            logger.warning("USB Camera was not explicitly closed")
            self.close()

    # Properties for compatibility with IDS Camera API

    @property
    def device(self):
        """Return mock device object for compatibility."""
        return self

    def ModelName(self) -> str:
        """Return model name (for compatibility)."""
        return self._model_name

    @property
    def acquisition_running(self) -> bool:
        """Check if acquisition is running."""
        return self._acquisition_running

    @property
    def width(self) -> int:
        """Get frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Get frame height."""
        return self._height

    @property
    def fps(self) -> float:
        """Get frame rate."""
        return self._fps


class USBImageView:
    """
    Wrapper for USB camera frame to provide IDS Peak ImageView-like interface.
    """

    def __init__(self, frame: np.ndarray):
        """
        Initialize with OpenCV frame (BGR).

        Args:
            frame: OpenCV frame (H, W, 3) in BGR format
        """
        self._frame = frame

    def get_numpy_array(self) -> np.ndarray:
        """
        Get frame as numpy array (BGR).

        Returns:
            Frame as numpy array (H, W, 3) in BGR format
        """
        return self._frame


class USBBuffer:
    """
    Wrapper for USB camera frame to provide IDS Peak Buffer-like interface.
    """

    def __init__(self, frame: np.ndarray):
        """
        Initialize with OpenCV frame.

        Args:
            frame: OpenCV frame (H, W, 3)
        """
        self._frame = frame
        self._incomplete = False

    def IsIncomplete(self) -> bool:
        """Check if buffer is incomplete (always False for USB cameras)."""
        return self._incomplete

    def Width(self) -> int:
        """Get frame width."""
        return self._frame.shape[1]

    def Height(self) -> int:
        """Get frame height."""
        return self._frame.shape[0]


class TimeoutException(Exception):
    """Exception raised on camera timeout (for compatibility with IDS Peak)."""
    pass
