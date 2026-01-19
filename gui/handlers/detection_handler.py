"""Detection handler for live anomaly detection workflow."""

from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Slot, QTimer

from live_anomaly_detector import LiveAnomalyDetector
from gui.threads import InferenceThread
from app_state import AppState


class DetectionHandler:
    """Handles live anomaly detection initialization and control."""

    def __init__(self, host):
        self.host = host
        # Debounce timer for saving runtime settings (avoid excessive disk I/O)
        self._save_timer: QTimer | None = None
        self._save_delay_ms = 500  # Save 500ms after last change

    def init_live_detection(self, memory_bank):
        """Initialize live anomaly detection."""
        try:
            import torch

            config = self.host.project_manager.current_config
            print(f"[DEBUG] Initializing live detection with model: {config.model_name}")

            # Coreset configuration
            CORESET_METHOD = "random"
            CORESET_RATIO = 0.1
            USE_CORESET = True

            self.host.anomaly_detector = LiveAnomalyDetector(
                model_name=config.model_name,
                reference_path=None,
                shots=config.shots,
                knn_k=config.knn_k,
                metric=config.metric,
                selected_layers=config.selected_layers,
                training_resolution=config.image_size,
                use_adaptive_knn=False,
                use_spatial_smoothing=False,
                use_faiss=True,
                use_coreset=USE_CORESET,
                coreset_method=CORESET_METHOD,
                coreset_ratio=CORESET_RATIO,
                # Motion Detection Parameters
                enable_motion_filter=config.enable_motion_filter,
                motion_high_threshold=config.motion_high_threshold,
                motion_low_threshold=config.motion_low_threshold,
                motion_stabilization_time=config.motion_stabilization_time,
                motion_learning_time=config.motion_learning_time
            )

            # Set memory bank
            if memory_bank.dtype == torch.float16:
                memory_bank = memory_bank.float()
            self.host.anomaly_detector.memory_bank = memory_bank

            # Apply coreset if needed
            expected_full_size = config.shots * 1024
            current_size = memory_bank.shape[0]
            already_reduced = current_size < (expected_full_size * 0.8)

            if USE_CORESET and not already_reduced:
                self.host.anomaly_detector.apply_coreset(
                    method=CORESET_METHOD,
                    ratio=CORESET_RATIO,
                    patches_per_image=1024
                )

            self.host.anomaly_detector.prepare_memory_bank()

            # Calculate the original trained threshold (with 20% margin)
            original_threshold = float(config.threshold) * 1.20

            # Set marker on threshold slider showing the original trained threshold
            self.host.threshold_slider.set_marker_value(int(original_threshold * 1000))

            # Restore runtime settings if available, otherwise use defaults
            if config.runtime_threshold is not None:
                # Use saved user-adjusted threshold
                self.host.current_threshold = config.runtime_threshold
                print(f"[DEBUG] Restoring saved threshold: {config.runtime_threshold:.3f}")
            else:
                # First time: use original threshold
                self.host.current_threshold = original_threshold

            # Restore confidence from runtime settings
            self.host.current_confidence = config.runtime_confidence
            print(f"[DEBUG] Restoring saved confidence: {config.runtime_confidence:.2f}")

            self.host.detection_active = True

            print(f"[DEBUG] Setting threshold slider to {int(self.host.current_threshold * 1000)}", flush=True)
            self.host.threshold_slider.setValue(int(self.host.current_threshold * 1000))
            self.host.threshold_value_label.setText(f"{self.host.current_threshold:.3f}")

            print(f"[DEBUG] Setting confidence slider to {int(self.host.current_confidence * 1000)}", flush=True)
            self.host.confidence_slider.setValue(int(self.host.current_confidence * 1000))
            self.host.confidence_value_label.setText(f"{self.host.current_confidence:.2f}")

            print(f"[DEBUG] Setting app_state to LIVE_DETECTION...", flush=True)
            print(f"[DEBUG] AppState.LIVE_DETECTION = {AppState.LIVE_DETECTION}", flush=True)
            self.host.app_state = AppState.LIVE_DETECTION
            print(f"[DEBUG] app_state SET to: {self.host.app_state}", flush=True)
            print(f"[DEBUG] Calling update_instruction_ui()...", flush=True)
            self.host.update_instruction_ui()
            print(f"[DEBUG] update_instruction_ui() RETURNED", flush=True)

            # Restore motion filter state from runtime settings
            if self.host.anomaly_detector.motion_filter is not None:
                motion_filter_active = config.runtime_motion_filter_active
                self.host.anomaly_detector.motion_filter.enabled = motion_filter_active
                # Button logic: checked=True means ON (filter active)
                # Stylesheet: unchecked=green, checked=red (inverted visuals)
                # But the original code uses checked=ON, so we follow that
                self.host.motion_filter_button.setChecked(motion_filter_active)
                if motion_filter_active:
                    self.host.motion_filter_button.setText("MOTION-FILTER: ON")
                else:
                    self.host.motion_filter_button.setText("MOTION-FILTER: OFF")
                print(f"[DEBUG] Restored motion filter state: {'ON' if motion_filter_active else 'OFF'}")

            # Restore visualization mode from runtime settings
            viz_mode = config.runtime_visualization_mode
            if viz_mode == "intensity":
                self.host.variant_method = self.host.visualization_variants.variant_intensity_bbox_gpu
                self.host.viz_mode_button.setChecked(True)
                self.host.viz_mode_button.setText("VIZ: INTENSITY")
            else:
                self.host.variant_method = self.host.visualization_variants.variant_efficient_minimal_gpu
                self.host.viz_mode_button.setChecked(False)
                self.host.viz_mode_button.setText("VIZ: CLASSIC")
            print(f"[DEBUG] Restored visualization mode: {viz_mode.upper()}")

            # Start inference thread
            if self.host.inference_thread is None:
                self.host.inference_thread = InferenceThread()
                self.host.inference_thread.inference_ready.connect(self.host.camera_handler.on_inference_ready)
                self.host.inference_thread.start()

            self.host.inference_thread.set_detector(
                detector=self.host.anomaly_detector,
                variant_method=self.host.variant_method,
                threshold=self.host.current_threshold,
                overlay_alpha=self.host._overlay_alpha,
                confidence=self.host.current_confidence
            )
            self.host.inference_thread.resume_processing()

            print("[OK] Live anomaly detection active!")

        except Exception as e:
            QMessageBox.critical(self.host, "Initialization Error", str(e))

    def toggle_detection(self):
        """Toggle anomaly detection on/off."""
        if self.host.app_state != AppState.LIVE_DETECTION:
            return

        self.host.detection_active = not self.host.detection_active

        if self.host.detection_active:
            self.host.action_button.setText("STOP")
            if self.host.inference_thread is not None:
                self.host.inference_thread.resume_processing()
        else:
            self.host.action_button.setText("START")
            if self.host.inference_thread is not None:
                self.host.inference_thread.pause_processing()
            self.host.last_inference_result = None
            self.host.camera_handler.last_inference_time_ms = 0.0
            self.host.camera_handler.last_index_search_time_ms = 0.0
            self.host.camera_handler.last_visualization_time_ms = 0.0
            self.host.set_stream_border_color("#1a1a1a", width=3)

    def on_threshold_change(self, value: int):
        """Handle threshold slider changes."""
        self.host.current_threshold = value / 1000.0
        self.host.threshold_value_label.setText(f"{self.host.current_threshold:.3f}")

        if self.host.inference_thread is not None:
            self.host.inference_thread.set_detector(
                detector=self.host.anomaly_detector,
                variant_method=self.host.variant_method,
                threshold=self.host.current_threshold,
                overlay_alpha=self.host._overlay_alpha,
                confidence=self.host.current_confidence
            )

        # Save runtime threshold to project config
        self._save_runtime_settings()

    def on_confidence_change(self, value: int):
        """Handle confidence slider changes."""
        self.host.current_confidence = value / 1000.0
        self.host.confidence_value_label.setText(f"{self.host.current_confidence:.2f}")

        if self.host.inference_thread is not None:
            self.host.inference_thread.set_detector(
                detector=self.host.anomaly_detector,
                variant_method=self.host.variant_method,
                threshold=self.host.current_threshold,
                overlay_alpha=self.host._overlay_alpha,
                confidence=self.host.current_confidence
            )

        # Save runtime confidence to project config
        self._save_runtime_settings()

    def toggle_motion_filter(self):
        """
        Toggle Motion Filter ON/OFF.

        ON (checked): Motion filter enabled, skips detection during motion (current behavior)
        OFF (unchecked): Motion filter disabled, runs detection continuously in real-time
        """
        if self.host.app_state != AppState.LIVE_DETECTION:
            return

        if self.host.anomaly_detector is None or self.host.anomaly_detector.motion_filter is None:
            print("[WARN] Motion filter not available (not enabled in config)")
            return

        # Toggle button state (checked = ON, unchecked = OFF)
        is_enabled = self.host.motion_filter_button.isChecked()

        if is_enabled:
            # Motion Filter ON: Enable motion detection (skip frames during motion)
            self.host.anomaly_detector.motion_filter.enabled = True
            self.host.motion_filter_button.setText("MOTION-FILTER: ON")
            print("[OK] Motion filter ENABLED - Detection pauses during motion")
        else:
            # Motion Filter OFF: Disable motion detection (run detection continuously)
            self.host.anomaly_detector.motion_filter.enabled = False
            self.host.motion_filter_button.setText("MOTION-FILTER: OFF")
            print("[OK] Motion filter DISABLED - Detection runs continuously in real-time")

        # Save motion filter state to project config
        self._save_runtime_settings()

    def toggle_visualization_mode(self):
        """
        Toggle visualization mode between CLASSIC (red fill) and INTENSITY (yellow→red gradient + bbox).

        CLASSIC (unchecked): Red alpha fill with red outline
        INTENSITY (checked): Intensity-based yellow→red coloring with bounding boxes
        """
        if self.host.app_state != AppState.LIVE_DETECTION:
            return

        # Toggle button state (unchecked = CLASSIC, checked = INTENSITY)
        is_intensity = self.host.viz_mode_button.isChecked()

        if is_intensity:
            # INTENSITY mode: Yellow→Red gradient with BBox
            self.host.variant_method = self.host.visualization_variants.variant_intensity_bbox_gpu
            self.host.viz_mode_button.setText("VIZ: INTENSITY")
            print("[OK] Visualization: INTENSITY mode (yellow→red gradient + bbox)")
        else:
            # CLASSIC mode: Red alpha fill
            self.host.variant_method = self.host.visualization_variants.variant_efficient_minimal_gpu
            self.host.viz_mode_button.setText("VIZ: CLASSIC")
            print("[OK] Visualization: CLASSIC mode (red alpha fill)")

        # Update inference thread with new visualization method
        if self.host.inference_thread is not None:
            self.host.inference_thread.set_detector(
                detector=self.host.anomaly_detector,
                variant_method=self.host.variant_method,
                threshold=self.host.current_threshold,
                overlay_alpha=self.host._overlay_alpha,
                confidence=self.host.current_confidence
            )

        # Save visualization mode to project config
        self._save_runtime_settings()

    def _save_runtime_settings(self):
        """
        Schedule saving of runtime settings with debounce.

        Uses a timer to avoid excessive disk I/O when slider is being dragged.
        Settings are saved 500ms after the last change.
        """
        if self.host.project_manager.current_config is None:
            return

        # Update config immediately (in memory)
        config = self.host.project_manager.current_config
        config.runtime_threshold = self.host.current_threshold
        config.runtime_confidence = self.host.current_confidence

        # Determine motion filter state
        if self.host.anomaly_detector is not None and self.host.anomaly_detector.motion_filter is not None:
            config.runtime_motion_filter_active = self.host.anomaly_detector.motion_filter.enabled
        else:
            # If no motion filter, preserve the button state (checked = ON)
            config.runtime_motion_filter_active = self.host.motion_filter_button.isChecked()

        # Determine visualization mode (unchecked = classic, checked = intensity)
        config.runtime_visualization_mode = "intensity" if self.host.viz_mode_button.isChecked() else "classic"

        # Debounce: restart timer on each call
        if self._save_timer is None:
            self._save_timer = QTimer()
            self._save_timer.setSingleShot(True)
            self._save_timer.timeout.connect(self._do_save_runtime_settings)

        self._save_timer.start(self._save_delay_ms)

    def _do_save_runtime_settings(self):
        """Actually save runtime settings to disk (called after debounce delay)."""
        if self.host.project_manager.current_config is None:
            return

        config = self.host.project_manager.current_config

        try:
            self.host.project_manager.save_config()
            print(f"[DEBUG] Runtime settings saved: threshold={config.runtime_threshold:.3f}, "
                  f"confidence={config.runtime_confidence:.2f}, "
                  f"motion_filter={'ON' if config.runtime_motion_filter_active else 'OFF'}, "
                  f"viz_mode={config.runtime_visualization_mode}")
        except Exception as e:
            print(f"[WARN] Failed to save runtime settings: {e}")
