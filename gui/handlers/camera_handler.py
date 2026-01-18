"""
Camera handler for initialization and frame management.

Handles:
    - Camera initialization and configuration (IDS Peak and USB)
    - Frame acquisition and processing
    - Display updates with anomaly overlays
    - FPS tracking
"""

from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from PySide6.QtWidgets import QApplication, QMessageBox, QDialog
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QPixmap

from ids_peak_common import CommonException
from ids_peak_icv.pipeline import DefaultPipeline

from camera import Camera
from gui.utils import image_to_numpy_rgb, numpy_to_qimage


class CameraHandler:
    """
    Handles camera initialization, frame acquisition, and display updates.

    Requirements:
        - Host must have: camera, pipeline, camera_thread, last_frame attributes
        - Host must have: image_widget, status_label, fps_label widgets
        - Host must have: app_state, detection_active, anomaly_detector, inference_thread
        - Host must have: set_stream_border_color(), animate_overlay_alpha() methods
    """

    def __init__(self, host):
        """
        Initialize camera handler.

        Args:
            host: Parent window (AnomalyDetectionAppQt instance)
        """
        self.host = host

        # Camera type ('ids' or 'usb')
        self.camera_type = None

        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.last_inference_time_ms = 0.0
        self.last_index_search_time_ms = 0.0
        self.last_visualization_time_ms = 0.0

        # Motion detection status
        self.motion_active = False
        self.motion_amount = 0.0
        self.detector_fps = 0.0

    def init_camera(self, camera_type: str = None):
        """
        Initialize camera (called after window is shown).

        Args:
            camera_type: 'ids' or 'usb' (if None, will prompt user)
        """
        try:
            print("=" * 60)
            print("Initializing camera...")
            print("=" * 60)

            # If camera_type not specified, show selection dialog
            if camera_type is None:
                from gui.dialogs import CameraTypeSelectionDialog
                dialog = CameraTypeSelectionDialog(self.host)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    camera_type = dialog.camera_type
                else:
                    # User cancelled
                    print("[WARN] Camera selection cancelled by user")
                    QApplication.quit()
                    return

            self.camera_type = camera_type
            print(f"[INFO] Selected camera type: {camera_type}")

            if camera_type == "ids":
                self._init_ids_camera()
            elif camera_type == "usb":
                self._init_usb_camera()
            else:
                raise ValueError(f"Unknown camera type: {camera_type}")

            # Start camera thread (works for both types)
            self._start_camera_thread()

            # Show project selection dialog
            QTimer.singleShot(500, self.host.project_handler.show_project_selection_dialog)

        except Exception as e:
            print(f"[FATAL] Camera initialization failed: {e}")
            self.show_fatal_camera_error(str(e))

    def _init_ids_camera(self):
        """Initialize IDS Peak camera."""
        print("[INFO] Initializing IDS Peak camera...")

        # GigE cameras: minimal initialization to avoid crashes
        self.host.camera = Camera(userset="Default", apply_optimizations=False, max_framerate=False)

        # Load camera settings from .cset file if exists
        cset_path = Path(__file__).parent.parent.parent / "camera_setup.cset"
        if cset_path.exists():
            print(f"Loading camera settings from {cset_path}")
            self.host.camera.load_settings_file(str(cset_path))
            print("[OK] Camera settings loaded from .cset file")
        else:
            print(f"[WARN] Warning: {cset_path} not found, using Default UserSet")

        self.host.camera.start_acquisition()
        self.host.setWindowTitle(f"Anomaly Detection - {self.host.camera.device.ModelName()}")
        self.host.pipeline = DefaultPipeline()

        self.host.status_label.setText(f"Kamera: {self.host.camera.device.ModelName()}")

        print("[OK] IDS Peak camera connected and streaming started!")

    def _init_usb_camera(self):
        """Initialize USB camera."""
        print("[INFO] Initializing USB camera...")

        from usb_camera import USBCamera

        # Initialize USB camera with auto-detection
        self.host.camera = USBCamera(
            camera_index=0,
            backend=cv2.CAP_MSMF,
            preferred_resolution=None,  # Auto-detect best resolution
            preferred_fps=None  # Auto-detect best FPS
        )

        self.host.camera.start_acquisition()
        self.host.setWindowTitle(f"Anomaly Detection - {self.host.camera.ModelName()}")
        self.host.pipeline = None  # No pipeline needed for USB cameras

        self.host.status_label.setText(f"Kamera: {self.host.camera.ModelName()} ({self.host.camera.width}x{self.host.camera.height})")

        print(f"[OK] USB camera connected: {self.host.camera.width}x{self.host.camera.height} @{self.host.camera.fps:.1f} FPS")

    def _start_camera_thread(self):
        """Start camera thread (works for both IDS Peak and USB)."""
        if self.camera_type == "ids":
            from gui.threads import CameraThread
            self.host.camera_thread = CameraThread(self.host.camera)
        elif self.camera_type == "usb":
            from gui.threads import USBCameraThread
            self.host.camera_thread = USBCameraThread(self.host.camera)
        else:
            raise ValueError(f"Unknown camera type: {self.camera_type}")

        self.host.camera_thread.frame_ready.connect(self.on_frame_ready)
        self.host.camera_thread.start()
        print(f"[OK] Camera thread started ({self.camera_type})")

    def show_fatal_camera_error(self, error_msg: str):
        """Show fatal camera error and exit."""
        QMessageBox.critical(
            self.host,
            "Camera Initialization Failed",
            f"Could not connect to IDS camera:\n\n{error_msg}\n\n"
            "Possible reasons:\n"
            "* No IDS camera connected\n"
            "* Camera in use by another application\n"
            "* IDS Peak SDK not installed\n\n"
            "Application will exit."
        )
        QApplication.quit()

    @Slot(object, object)
    def on_frame_ready(self, image_view, buffer):
        """Handle frame ready signal from camera thread (works for both IDS Peak and USB)."""
        try:
            # Process image (different for IDS Peak and USB)
            if self.camera_type == "ids":
                # IDS Peak: use pipeline
                processed_img = self.host.pipeline.process(image_view)
            else:
                # USB: image_view is already the processed frame (USBImageView)
                processed_img = image_view

            self.host.last_frame = processed_img

            # Increment frame count (actual camera frames, not display updates)
            self.frame_count += 1

            # Import AppState
            from app_state import AppState

            # Submit frame for async inference if in detection mode
            if (self.host.app_state == AppState.LIVE_DETECTION and
                self.host.anomaly_detector is not None and
                self.host.detection_active and
                self.host.inference_thread is not None):

                # Convert to numpy
                img_np = image_to_numpy_rgb(processed_img)

                # Submit to inference thread (non-blocking)
                self.host.inference_thread.submit_frame(img_np)

            # Queue buffer back
            self.host.camera.queue_buffer(buffer)

        except CommonException:
            self.host.camera.queue_buffer(buffer)
        except Exception as e:
            # Generic exception for USB cameras
            print(f"[ERROR] Frame processing error: {e}")
            self.host.camera.queue_buffer(buffer)

    def update_image(self):
        """Update image display (30 FPS) - uses async inference results."""
        if self.host.last_frame is None:
            return

        try:
            # Import AppState
            from app_state import AppState

            # SKIP camera update during text overlay intro states
            # (text overlay is shown in image_widget, camera would overwrite it)
            if self.host.app_state in [
                AppState.SHOW_STEP_1_INTRO,
                AppState.SHOW_STEP_2_INTRO,
                AppState.SHOW_STEP_3_INTRO
            ]:
                return  # Don't update image_widget during text overlay

            # Convert to numpy
            img_np = image_to_numpy_rgb(self.host.last_frame)

            # Use latest inference result (ONLY if detection is active)
            if (self.host.app_state == AppState.LIVE_DETECTION and
                self.host.detection_active and
                self.host.last_inference_result is not None):

                anomaly_map, final_img, max_score = self.host.last_inference_result

                # Store results with moving average (5 frames) for smoother reaction
                self.host.anomaly_score_window.append(max_score)
                smoothed_score = sum(self.host.anomaly_score_window) / len(self.host.anomaly_score_window)
                self.host.last_anomaly_score = smoothed_score
                self.host.is_anomaly_detected = (smoothed_score > self.host.current_threshold)

                # Smooth overlay alpha animation based on anomaly detection
                if self.host.is_anomaly_detected:
                    # Fast fade in for anomaly detection (immediate warning)
                    self.host.animate_overlay_alpha(self.host._target_overlay_alpha, duration=100)
                else:
                    # Fast fade out when score < threshold
                    self.host.animate_overlay_alpha(0.0, duration=300)

                # Update status with scores
                self.host.status_label.setText(
                    f"Projekt: {self.host.project_manager.current_project} | "
                    f"Score Ø5: {smoothed_score:.4f} | Schwelle: {self.host.current_threshold:.4f}"
                )

                # Stream border color feedback - instant (no animation)
                if self.motion_active:
                    # Motion detected - neutral border
                    self.host.set_stream_border_color("#1a1a1a", width=3)  # dark neutral border
                elif self.host.is_anomaly_detected:
                    # Anomaly detected - RED border
                    self.host.set_stream_border_color("#ff0000", width=8)  # red border (anomaly detected)
                else:
                    # OK - GREEN border
                    self.host.set_stream_border_color("#00ff00", width=8)  # green border (OK)

            else:
                # No inference active or no result - show raw live image
                final_img = img_np

                # Reset border to neutral when detection is stopped
                if self.host.app_state == AppState.LIVE_DETECTION and not self.host.detection_active:
                    # Detection is paused - show neutral border
                    self.host.set_stream_border_color("#505050", width=3)  # neutral gray border
                    # Update status to show paused state
                    self.host.status_label.setText(
                        f"Projekt: {self.host.project_manager.current_project} | Detektion pausiert"
                    )
                elif self.host.app_state != AppState.LIVE_DETECTION:
                    self.host.set_stream_border_color("#1a1a1a", width=3)  # dark neutral border

            # Convert to QPixmap and display
            qimage = numpy_to_qimage(final_img)
            pixmap = QPixmap.fromImage(qimage)

            # Scale to fit widget while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.host.image_widget.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.host.image_widget.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"[ERROR] Update loop: {e}")
            import traceback
            traceback.print_exc()

    @Slot(object, object, float, float, float, float, bool, float, float)
    def on_inference_ready(
        self,
        anomaly_map,
        final_img,
        max_score,
        inference_time_ms,
        index_search_time_ms,
        visualization_time_ms,
        motion_active,
        motion_amount,
        current_fps
    ):
        """Handle inference result from background thread (non-blocking)."""
        # Store latest inference result for display in update_image()
        self.host.last_inference_result = (anomaly_map, final_img, max_score)

        # Store performance metrics
        self.last_inference_time_ms = inference_time_ms
        self.last_index_search_time_ms = index_search_time_ms
        self.last_visualization_time_ms = visualization_time_ms

        # Store motion detection status
        self.motion_active = motion_active
        self.motion_amount = motion_amount
        self.detector_fps = current_fps

    def update_fps(self):
        """Update FPS counter and performance metrics."""
        self.fps = self.frame_count

        # Build multi-line performance text with FIXED number of lines
        # Start with motion status
        if self.motion_active:
            perf_text = f"[WARN] MOTION: {self.motion_amount*100:.1f}%"
        elif self.motion_amount > 0:
            perf_text = f"[OK] Static: {self.motion_amount*100:.1f}%"
        else:
            perf_text = "Status: Ready"

        # Add FPS (camera and detector) - ALWAYS show
        perf_text += f"\nFPS: {self.fps:.1f}"
        if self.detector_fps > 0:
            perf_text += f" (Detector: {self.detector_fps:.1f})"
        else:
            perf_text += " (Detector: ---)"

        # Show inference timing - ALWAYS show all metrics (use --- if zero)
        if self.last_inference_time_ms > 0:
            perf_text += f"\nInferenz (Total): {self.last_inference_time_ms:.1f}ms"
        else:
            perf_text += f"\nInferenz (Total): ---"

        if self.last_index_search_time_ms > 0:
            perf_text += f"\nIndex-Suche: {self.last_index_search_time_ms:.1f}ms"
        else:
            perf_text += f"\nIndex-Suche: ---"

        if self.last_visualization_time_ms > 0:
            perf_text += f"\nVisualisierung: {self.last_visualization_time_ms:.1f}ms"
        else:
            perf_text += f"\nVisualisierung: ---"

        self.host.fps_label.setText(perf_text)
        self.frame_count = 0
