"""Detection handler for live anomaly detection workflow."""

from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Slot

from live_anomaly_detector import LiveAnomalyDetector
from gui.threads import InferenceThread
from app_state import AppState


class DetectionHandler:
    """Handles live anomaly detection initialization and control."""

    def __init__(self, host):
        self.host = host

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
            # Set threshold 5% higher than optimized value (more robust in real conditions)
            self.host.current_threshold = float(config.threshold) * 1.05
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
