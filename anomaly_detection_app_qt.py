#!/usr/bin/env python3
"""
IDS Camera Anomaly Detection App (PySide6)
==========================================

Complete workflow for anomaly detection:
1. Project selection (load existing or create new)
2. Data collection (30 train/test good images + 20 defect images)
3. Memory bank training
4. Threshold optimization
5. Live anomaly detection

Usage:
    python anomaly_detection_app_qt.py
"""

import sys
from collections import deque
from pathlib import Path
from typing import Optional
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QKeyEvent, QPixmap

# IDS Peak SDK
from ids_peak_icv import Image
from ids_peak_icv.pipeline import DefaultPipeline

# Backend modules
from camera import Camera
from project_manager import ProjectManager
from live_anomaly_detector import LiveAnomalyDetector
from visualization_variants import VisualizationVariants
from app_state import AppState

# GUI modules - Refactored architecture
from gui import (
    # Threads
    TrainingThread, InferenceThread, CameraThread,
    # Utilities
    image_to_numpy_rgb, numpy_to_qimage,
    # Visual Effects Mixin
    VisualEffectsMixin,
    # UI Builder
    UIBuilder,
    # Handlers
    CameraHandler, DetectionHandler, TrainingHandler,
    ProjectHandler, CaptureHandler
)


class AnomalyDetectionAppQt(QMainWindow, VisualEffectsMixin):
    """
    Main application window (PySide6).

    Architecture:
        - Inherits from VisualEffectsMixin for animations
        - Uses composition pattern with specialized handlers
        - Delegates UI building to UIBuilder

    Handlers:
        - CameraHandler: Camera initialization and frame management
        - DetectionHandler: Live anomaly detection workflow
        - TrainingHandler: Model training coordination
        - ProjectHandler: Project selection and management
        - CaptureHandler: Image capture workflows
    """

    def __init__(self):
        super().__init__()

        print("[DEBUG] Initializing Qt Application...")

        # Window configuration
        self.setWindowTitle("Anomaly Detection")
        self.setMinimumSize(1280, 1024)

        # Initialize visual effects mixin
        self.init_visual_effects()

        # Camera and pipeline (initialized after window is ready)
        self.camera: Optional[Camera] = None
        self.pipeline: Optional[DefaultPipeline] = None
        self.last_frame: Optional[Image] = None
        self.camera_thread: Optional[CameraThread] = None

        # Project management
        self.project_manager = ProjectManager()
        self.anomaly_detector: Optional[LiveAnomalyDetector] = None

        # State management
        self.app_state = AppState.PROJECT_SELECTION
        self.captured_images = []
        self.capture_target = 0
        self.capture_category = ""

        # Anomaly detection state
        self.current_threshold = 0.5
        self.current_confidence = 0.05  # Very low default for edge detection (minimal noise)
        self.detection_active = False
        self.last_anomaly_score = 0.0
        self.is_anomaly_detected = False
        self.anomaly_score_window = deque(maxlen=5)

        # Visualization (GPU-accelerated variant)
        self.visualization_variants = VisualizationVariants()
        self.variant_method = self.visualization_variants.variant_efficient_minimal_gpu

        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0

        # Threads
        self.training_thread: Optional[TrainingThread] = None
        self.inference_thread: Optional[InferenceThread] = None
        self.last_inference_result = None

        # Initialize handlers (composition pattern)
        self.camera_handler = CameraHandler(self)
        self.detection_handler = DetectionHandler(self)
        self.training_handler = TrainingHandler(self)
        self.project_handler = ProjectHandler(self)
        self.capture_handler = CaptureHandler(self)

        # Build UI (delegate to UIBuilder)
        UIBuilder.build_main_ui(self)
        UIBuilder.apply_dark_theme(self)
        UIBuilder.setup_timers(self)
        self.setup_controls_auto_hide()

        # Text overlay intro timer (3s auto-advance)
        self.intro_timer = QTimer(self)
        self.intro_timer.setSingleShot(True)
        self.intro_timer.setInterval(3000)  # 3 seconds
        self.intro_timer.timeout.connect(self.advance_from_intro)

        # Set initial neutral border for livestream
        self.set_stream_border_color("#505050", width=3)

        print("[DEBUG] Qt Application initialized")

    # ========== Camera Management ==========

    def init_camera(self):
        """Initialize camera (delegated to CameraHandler)."""
        self.camera_handler.init_camera()

    # ========== UI Updates ==========

    def update_instruction_ui(self):
        """Update instruction label and button based on current state."""
        try:
            print(f"[DEBUG] update_instruction_ui() CALLED", flush=True)
            print(f"[DEBUG] app_state = {self.app_state}", flush=True)
            print(f"[DEBUG] app_state type: {type(self.app_state)}", flush=True)
            print(f"[DEBUG] app_state.value: {self.app_state.value}", flush=True)
            print(f"[DEBUG] AppState.LIVE_DETECTION = {AppState.LIVE_DETECTION}", flush=True)
            print(f"[DEBUG] Match LIVE_DETECTION? {self.app_state == AppState.LIVE_DETECTION}", flush=True)
        except Exception as e:
            print(f"[ERROR] Debug print failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

        if self.app_state == AppState.PROJECT_SELECTION:
            if self.project_manager.current_project:
                self.instruction_label.setText(
                    f"<b>Projekt: {self.project_manager.current_project}</b><br>"
                    "Bereit für Bildaufnahme-Workflow"
                )
                self.instruction_label.setStyleSheet("font-size: 18px; color: #e0e0e0;")
                self.action_button.setText("AUFNAHME STARTEN")
                self.action_button.setEnabled(True)
                self.status_text.setText("")
            else:
                self.instruction_label.setText("Warte auf Projektauswahl...")
                self.instruction_label.setStyleSheet("font-size: 18px; color: #e0e0e0;")
                self.action_button.setEnabled(False)
                self.status_text.setText("")

            self.progress_bar.setValue(0)
            self.progress_bar.hide()
            self.progress_label.setText("")
            self.threshold_slider.hide()
            self.threshold_label.hide()
            self.threshold_value_label.hide()
            self.confidence_slider.hide()
            self.confidence_label.hide()
            self.confidence_value_label.hide()
            self.undo_button.hide()

        # ========== Text Overlay Intros (3s black screen with white text) ==========

        elif self.app_state == AppState.SHOW_STEP_1_INTRO:
            # Display black screen with white text in video frame
            self.show_text_overlay(
                "SCHRITT 1/3\n"
                "5 Hintergrundbilder aufnehmen\n"
                "(ohne Objekt)"
            )
            self.instruction_label.setText("")
            self.instruction_label.setStyleSheet("font-size: 18px; color: #e0e0e0;")
            self.action_button.setEnabled(False)
            self.progress_bar.hide()
            self.progress_label.setText("")
            self.status_text.setText("")
            self.threshold_slider.hide()
            self.threshold_label.hide()
            self.threshold_value_label.hide()
            self.confidence_slider.hide()
            self.confidence_label.hide()
            self.confidence_value_label.hide()
            self.undo_button.hide()
            # Start 3s timer for auto-advance
            self.intro_timer.start()

        elif self.app_state == AppState.SHOW_STEP_2_INTRO:
            config = self.project_manager.current_config
            total_good = (config.train_target + config.test_good_target) if config else 0
            # Display black screen with white text in video frame
            self.show_text_overlay(
                "SCHRITT 2/3\n"
                f"{total_good} Bilder von Zustand GUT aufnehmen"
            )
            self.instruction_label.setText("")
            self.instruction_label.setStyleSheet("font-size: 18px; color: #e0e0e0;")
            self.action_button.setEnabled(False)
            self.progress_bar.hide()
            self.progress_label.setText("")
            self.status_text.setText("")
            self.threshold_slider.hide()
            self.threshold_label.hide()
            self.threshold_value_label.hide()
            self.confidence_slider.hide()
            self.confidence_label.hide()
            self.confidence_value_label.hide()
            self.undo_button.hide()
            # Start 3s timer for auto-advance
            self.intro_timer.start()

        elif self.app_state == AppState.SHOW_STEP_3_INTRO:
            config = self.project_manager.current_config
            defect_count = config.test_defect_target if config else 0
            # Display black screen with white text in video frame
            self.show_text_overlay(
                "SCHRITT 3/3\n"
                f"{defect_count} Bilder von Zustand NOK aufnehmen"
            )
            self.instruction_label.setText("")
            self.instruction_label.setStyleSheet("font-size: 18px; color: #e0e0e0;")
            self.action_button.setEnabled(False)
            self.progress_bar.hide()
            self.progress_label.setText("")
            self.status_text.setText("")
            self.threshold_slider.hide()
            self.threshold_label.hide()
            self.threshold_value_label.hide()
            self.confidence_slider.hide()
            self.confidence_label.hide()
            self.confidence_value_label.hide()
            self.undo_button.hide()
            # Start 3s timer for auto-advance
            self.intro_timer.start()

        # ========== Capture States ==========

        elif self.app_state == AppState.CAPTURE_BACKGROUND:
            # STEP 1: Background baseline capture (always 5 images, no object present)
            # Live camera feed resumes automatically
            self.instruction_label.setText(
                f"<b>SCHRITT 1/3: {self.capture_target} HINTERGRUNDBILDER</b><br>"
                "NUR UNTERGRUND/HINTERGRUND - KEIN OBJEKT!"
            )
            self.instruction_label.setStyleSheet("font-size: 18px; color: #ff9944;")
            self.action_button.setText("AUFNEHMEN")
            self.action_button.setEnabled(True)
            self.progress_bar.setMaximum(self.capture_target)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
            self.progress_label.setText("")
            self.status_text.setText(
                "Diese Bilder werden verwendet, um die Baseline für 'kein Objekt vorhanden' zu berechnen.\n"
                "Wichtig: Nur den leeren Untergrund aufnehmen, OHNE Objekt im Bild!"
            )
            self.threshold_slider.hide()
            self.threshold_label.hide()
            self.threshold_value_label.hide()
            self.confidence_slider.hide()
            self.confidence_label.hide()
            self.confidence_value_label.hide()
            self.undo_button.show()
            self.undo_button.setEnabled(len(self.captured_images) > 0)
            # Ensure controls are visible
            self.controls_panel.show()
            self.toggle_controls_button.setText("▼ Controls ausblenden")
            self.reset_controls_auto_hide_timer()

        elif self.app_state == AppState.CAPTURE_GOOD:
            # STEP 2: Good images capture (train + test merged, split automatically)
            # Live camera feed resumes automatically
            config = self.project_manager.current_config
            total_good = (config.train_target + config.test_good_target) if config else 0
            self.instruction_label.setText(
                f"<b>SCHRITT 2/3: {total_good} GUTE TEILE</b><br>"
                "AUFNEHMEN klicken oder LEERTASTE"
            )
            self.instruction_label.setStyleSheet("font-size: 18px; color: #44ff44;")
            self.action_button.setText("AUFNEHMEN")
            self.action_button.setEnabled(True)
            self.progress_bar.setMaximum(total_good)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
            self.progress_label.setText("")
            self.status_text.setText(
                "Gute (fehlerfreie) Teile im Frame platzieren.\n"
                "Diese Bilder werden für Training und Validierung verwendet."
            )
            self.threshold_slider.hide()
            self.threshold_label.hide()
            self.threshold_value_label.hide()
            self.confidence_slider.hide()
            self.confidence_label.hide()
            self.confidence_value_label.hide()
            self.undo_button.show()
            self.undo_button.setEnabled(len(self.captured_images) > 0)
            # Ensure controls are visible
            self.controls_panel.show()
            self.toggle_controls_button.setText("▼ Controls ausblenden")
            self.reset_controls_auto_hide_timer()

        elif self.app_state == AppState.CAPTURE_TEST_DEFECT:
            # STEP 3: Defect images capture
            # Live camera feed resumes automatically
            self.instruction_label.setText(
                f"<b>SCHRITT 3/3: {self.capture_target} DEFEKTE TEILE</b><br>"
                "AUFNEHMEN klicken oder LEERTASTE"
            )
            self.instruction_label.setStyleSheet("font-size: 18px; color: #ff4444;")
            self.action_button.setText("AUFNEHMEN")
            self.action_button.setEnabled(True)
            self.progress_bar.setMaximum(self.capture_target)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
            self.progress_label.setText("")
            self.status_text.setText("Defekte Teile im Frame platzieren.\nBeispiele: Kratzer, Dellen, fehlende Teile, Verfärbungen")
            self.threshold_slider.hide()
            self.threshold_label.hide()
            self.threshold_value_label.hide()
            self.confidence_slider.hide()
            self.confidence_label.hide()
            self.confidence_value_label.hide()
            self.undo_button.show()
            self.undo_button.setEnabled(len(self.captured_images) > 0)
            # Ensure controls are visible
            self.controls_panel.show()
            self.toggle_controls_button.setText("▼ Controls ausblenden")
            self.reset_controls_auto_hide_timer()

        elif self.app_state == AppState.TRAINING:
            self.instruction_label.setText("Training läuft...")
            self.instruction_label.setStyleSheet("font-size: 18px; color: #e0e0e0;")
            self.action_button.setEnabled(False)
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
            self.progress_label.setText("")
            self.status_text.setText("Bitte warten Sie, während das Modell trainiert wird.\nDies kann mehrere Minuten dauern.")
            self.threshold_slider.hide()
            self.threshold_label.hide()
            self.threshold_value_label.hide()
            self.confidence_slider.hide()
            self.confidence_label.hide()
            self.confidence_value_label.hide()
            self.undo_button.hide()

        elif self.app_state == AppState.LIVE_DETECTION:
            print("[DEBUG] ========== LIVE_DETECTION STATE ==========", flush=True)
            print(f"[DEBUG] Setting up LIVE_DETECTION UI...", flush=True)

            try:
                summary = self.project_manager.get_project_summary()
                print(f"[DEBUG] Got project summary: {summary['name']}", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to get project summary: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return

            self.instruction_label.setText("Live Anomalie-Erkennung")
            self.instruction_label.setStyleSheet("font-size: 18px; color: #00ff88; font-weight: bold;")
            self.progress_bar.hide()
            self.progress_label.setText("")

            # Status in right column
            status_info = (
                f"<b>Projekt:</b> {summary['name']}<br>"
                f"<b>Genauigkeit:</b> {summary['validation']['accuracy']:.2%}<br>"
                f"<b>F1-Score:</b> {summary['validation']['f1_score']:.2%}<br>"
                f"<b>Visualisierung:</b> Optimiert (minimal)"
            )
            self.status_text.setText(status_info)

            # Show sliders
            print(f"[DEBUG] Showing sliders...", flush=True)
            print(f"[DEBUG] confidence_slider: {self.confidence_slider}", flush=True)
            print(f"[DEBUG] confidence_label: {self.confidence_label}", flush=True)
            print(f"[DEBUG] threshold_slider: {self.threshold_slider}", flush=True)
            print(f"[DEBUG] threshold_label: {self.threshold_label}", flush=True)

            self.confidence_slider.show()
            self.confidence_label.show()
            self.confidence_value_label.show()
            self.threshold_slider.show()
            self.threshold_label.show()
            self.threshold_value_label.show()

            print(f"[DEBUG] After show() - confidence_slider.isVisible(): {self.confidence_slider.isVisible()}", flush=True)
            print(f"[DEBUG] After show() - confidence_label.isVisible(): {self.confidence_label.isVisible()}", flush=True)
            print(f"[DEBUG] After show() - confidence_value_label.isVisible(): {self.confidence_value_label.isVisible()}", flush=True)
            print(f"[DEBUG] After show() - threshold_slider.isVisible(): {self.threshold_slider.isVisible()}", flush=True)
            print(f"[DEBUG] After show() - threshold_label.isVisible(): {self.threshold_label.isVisible()}", flush=True)
            print(f"[DEBUG] After show() - threshold_value_label.isVisible(): {self.threshold_value_label.isVisible()}", flush=True)

            # Configure button
            if self.detection_active:
                self.action_button.setText("STOP")
                print("[DEBUG] Button set to STOP (detection active)", flush=True)
            else:
                self.action_button.setText("START")
                print("[DEBUG] Button set to START (detection inactive)", flush=True)
            self.action_button.setEnabled(True)
            self.undo_button.hide()

            # Ensure controls are visible
            self.controls_panel.show()
            self.toggle_controls_button.setText("▼ Controls ausblenden")
            self.reset_controls_auto_hide_timer()

            print("[DEBUG] ========== LIVE_DETECTION UI COMPLETE ==========", flush=True)

    def setup_controls_auto_hide(self):
        """Configure automatic hiding for the controls panel."""
        self.controls_auto_hide_timer = QTimer(self)
        self.controls_auto_hide_timer.setInterval(6000)  # 6 seconds
        self.controls_auto_hide_timer.setSingleShot(True)
        self.controls_auto_hide_timer.timeout.connect(self.auto_hide_controls_panel)

        self.setMouseTracking(True)
        self.centralWidget().setMouseTracking(True)
        self.controls_panel.setMouseTracking(True)

        self.installEventFilter(self)
        self.centralWidget().installEventFilter(self)
        self.controls_panel.installEventFilter(self)
        self.toggle_controls_button.installEventFilter(self)

        self.reset_controls_auto_hide_timer()

    def reset_controls_auto_hide_timer(self):
        """Restart auto-hide timer when controls are in use."""
        if not self.controls_panel.isVisible():
            return

        # Auto-hide in all capture and detection states (maximize camera view)
        if self.app_state in [
            AppState.CAPTURE_BACKGROUND,
            AppState.CAPTURE_GOOD,
            AppState.CAPTURE_TEST_DEFECT,
            AppState.LIVE_DETECTION
        ]:
            # Restart timer - will hide controls after 6 seconds of inactivity
            self.controls_auto_hide_timer.start()
        else:
            # Keep controls visible in other states (PROJECT_SELECTION, TRAINING, etc.)
            self.controls_auto_hide_timer.stop()

    def auto_hide_controls_panel(self):
        """Hide controls after inactivity timeout."""
        if self.controls_panel.isVisible():
            self.controls_panel.hide()
            self.toggle_controls_button.setText("▲ Controls einblenden")

    def toggle_controls_panel(self):
        """Toggle controls panel visibility to maximize live view."""
        if self.controls_panel.isVisible():
            self.controls_panel.hide()
            self.toggle_controls_button.setText("▲ Controls einblenden")
            self.controls_auto_hide_timer.stop()
            print("[OK] Controls panel hidden - Live view maximized")
        else:
            self.controls_panel.show()
            self.toggle_controls_button.setText("▼ Controls ausblenden")
            self.reset_controls_auto_hide_timer()
            print("[OK] Controls panel shown")

    def eventFilter(self, watched, event):
        """Detect user activity to keep controls visible."""
        if self.controls_panel.isVisible() and event.type() in {
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.Wheel,
            QEvent.Type.KeyPress
        }:
            self.reset_controls_auto_hide_timer()
        return super().eventFilter(watched, event)

    def show_text_overlay(self, text: str):
        """
        Display black screen with white text in the image_widget.
        Used for step intro overlays to signal "something new is coming!"

        Args:
            text: Text to display (supports HTML formatting)
        """
        # Get current image_widget size
        width = self.image_widget.width()
        height = self.image_widget.height()

        # Ensure minimum size
        if width < 640:
            width = 640
        if height < 480:
            height = 480

        # Create black image
        import cv2
        black_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Convert text to plain text (remove HTML tags for OpenCV)
        import re
        plain_text = re.sub('<.*?>', '', text)
        plain_text = plain_text.replace('<br>', '\n')

        # Split into lines
        lines = plain_text.strip().split('\n')

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        color = (255, 255, 255)  # White

        # Calculate text sizes and positions
        line_heights = []
        line_widths = []
        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
            line_heights.append(text_height + baseline)
            line_widths.append(text_width)

        # Total height and spacing
        line_spacing = 20
        total_text_height = sum(line_heights) + line_spacing * (len(lines) - 1)

        # Starting Y position (centered vertically)
        y_start = (height - total_text_height) // 2

        # Draw each line centered horizontally
        y_pos = y_start
        for i, line in enumerate(lines):
            # Center horizontally
            x_pos = (width - line_widths[i]) // 2
            y_pos += line_heights[i]

            cv2.putText(black_img, line, (x_pos, y_pos), font, font_scale, color, font_thickness, cv2.LINE_AA)
            y_pos += line_spacing

        # Convert to QPixmap and display
        from gui.utils import numpy_to_qimage
        qimage = numpy_to_qimage(black_img)
        pixmap = QPixmap.fromImage(qimage)
        self.image_widget.setPixmap(pixmap)

        print(f"[DEBUG] Text overlay displayed: {lines[0] if lines else 'empty'}")

    def advance_from_intro(self):
        """
        Auto-advance from text overlay intro to corresponding capture state.
        Called by intro_timer after 3 seconds.
        """
        if self.app_state == AppState.SHOW_STEP_1_INTRO:
            # Advance to background capture
            self.app_state = AppState.CAPTURE_BACKGROUND
            self.update_instruction_ui()
            # Force immediate camera frame update to replace text overlay
            self.camera_handler.update_image()
        elif self.app_state == AppState.SHOW_STEP_2_INTRO:
            # Advance to good images capture
            self.app_state = AppState.CAPTURE_GOOD
            self.update_instruction_ui()
            # Force immediate camera frame update to replace text overlay
            self.camera_handler.update_image()
        elif self.app_state == AppState.SHOW_STEP_3_INTRO:
            # Advance to defect images capture
            self.app_state = AppState.CAPTURE_TEST_DEFECT
            self.update_instruction_ui()
            # Force immediate camera frame update to replace text overlay
            self.camera_handler.update_image()

    # ========== Event Handlers ==========

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events."""
        # Spacebar triggers capture
        if event.key() == Qt.Key.Key_Space:
            if self.app_state in [
                AppState.CAPTURE_BACKGROUND,
                AppState.CAPTURE_GOOD,  # Merged train+test good
                AppState.CAPTURE_TEST_DEFECT
            ]:
                self.capture_handler.capture_image()
                event.accept()
                return

        super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close event."""
        print("Shutting down...")

        # Stop inference thread
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread.wait(1000)
            print("  [OK] Inference thread stopped")

        # Stop camera thread
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait(1000)
            print("  [OK] Camera thread stopped")

        # Stop camera
        if self.camera is not None:
            try:
                self.camera.kill_datastream_wait()
                self.camera.stop_acquisition()
                del self.camera
                print("  [OK] Camera stopped")
            except Exception as e:
                print(f"[ERROR] Camera shutdown: {e}")

        event.accept()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Starting Anomaly Detection App (PySide6)...")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 8):
        print("[FATAL] Python 3.8+ required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)

    # Check critical imports
    print("\nChecking dependencies...")
    critical_deps = [
        ("PySide6", "PySide6"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
    ]

    missing = []
    for module, name in critical_deps:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} - NOT FOUND")
            missing.append(name)

    if missing:
        print(f"\n[FATAL] Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements_qt.txt")
        sys.exit(1)

    # Check IDS Peak SDK
    try:
        from ids_peak import ids_peak
        print("  [OK] IDS Peak SDK")
    except ImportError:
        print("  [FAIL] IDS Peak SDK not found!")
        print("\nDownload from: https://www.ids-imaging.com/downloads.html")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All checks passed! Starting app...")
    print("=" * 60 + "\n")

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Anomaly Detection")

    # Create main window
    window = AnomalyDetectionAppQt()
    window.showMaximized()

    # Initialize camera after window is shown
    QTimer.singleShot(100, window.init_camera)

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
