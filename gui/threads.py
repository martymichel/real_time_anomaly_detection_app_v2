"""
Background threading classes for GUI operations.

Classes:
    - TrainingThread: Background thread for model training
    - InferenceThread: Background thread for asynchronous anomaly detection
    - CameraThread: Background thread for camera frame acquisition
"""

import threading
import numpy as np
from PySide6.QtCore import QThread, Signal

# IDS Peak SDK
from ids_peak_common import CommonException
from ids_peak import ids_peak

# Backend modules
from model_trainer import ModelTrainer


class TrainingThread(QThread):
    """Background thread for model training."""

    progress_changed = Signal(str, float)  # message, value
    training_complete = Signal(object)  # memory_bank
    training_error = Signal(str)  # error message

    def __init__(self, project_manager, model_name, shots, batch_size, knn_k, metric):
        super().__init__()
        self.project_manager = project_manager
        self.model_name = model_name
        self.shots = shots
        self.batch_size = batch_size
        self.knn_k = knn_k
        self.metric = metric

    def run(self):
        """Run training in background."""
        try:
            config = self.project_manager.current_config

            # Initialize trainer
            self.progress_changed.emit("Loading model...", 0.1)
            model_trainer = ModelTrainer(
                model_name=config.model_name,
                device=None,  # Auto-detect
                cache_dir=None,
                selected_layers=config.selected_layers
            )
            model_trainer.load_model()

            # Build memory bank
            self.progress_changed.emit("Building memory bank...", 0.2)
            train_path = self.project_manager.get_train_images_path()

            def progress_callback(cur, total, msg):
                progress = 0.2 + 0.5 * (cur / total)
                self.progress_changed.emit(msg, progress)

            memory_bank = model_trainer.build_memory_bank(
                train_path=train_path,
                shots=config.shots,
                batch_size=config.batch_size,
                image_size=config.image_size,
                progress_callback=progress_callback
            )

            # Apply coreset reduction if configured
            if config.coreset_percentage is not None:
                def coreset_callback(cur, total, msg):
                    progress = 0.7 + 0.1 * (cur / total)
                    self.progress_changed.emit(msg, progress)

                self.progress_changed.emit(
                    f"Applying coreset reduction ({config.coreset_percentage}%)...",
                    0.7
                )

                memory_bank = model_trainer.reduce_memory_bank_coreset(
                    percentage=config.coreset_percentage,
                    method=config.coreset_method,
                    progress_callback=coreset_callback
                )

            # Save memory bank
            self.project_manager.save_memory_bank(memory_bank)

            # Optimize threshold with validation visualizations
            self.progress_changed.emit("Optimizing threshold...", 0.8)
            test_good_path, test_defect_path = self.project_manager.get_test_images_paths()

            def threshold_callback(cur, total, msg):
                progress = 0.8 + 0.15 * (cur / total)
                self.progress_changed.emit(msg, progress)

            # Create validation output directory
            project_path = self.project_manager.get_project_path(self.project_manager.current_project)
            validation_output_dir = project_path / "results" / "val"

            threshold, metrics = model_trainer.optimize_threshold(
                test_good_path=test_good_path,
                test_defect_path=test_defect_path,
                knn_k=config.knn_k,
                metric=config.metric,
                validation_output_dir=validation_output_dir,
                progress_callback=threshold_callback
            )

            # Save validation results
            self.progress_changed.emit("Saving validation results...", 0.95)
            self.project_manager.save_validation_results(
                threshold=threshold,
                **metrics
            )

            # Training complete
            self.progress_changed.emit("Training complete!", 1.0)
            self.training_complete.emit(memory_bank)

        except Exception as e:
            self.training_error.emit(str(e))


class InferenceThread(QThread):
    """Background thread for asynchronous anomaly detection inference."""

    inference_ready = Signal(object, object, float, float, float, float, bool, float, float)  # anomaly_map, final_img, max_score, total_inference_ms, index_search_ms, visualization_time_ms, motion_active, motion_amount, current_fps

    def __init__(self):
        super().__init__()
        self.running = True
        self.processing_enabled = True  # Flag to pause/resume processing
        self.detector = None
        self.variant_method = None
        self.threshold = 0.5
        self.overlay_alpha = 0.5
        self.confidence = 0.5
        self.pending_frame = None
        self.frame_lock = threading.Lock()
        self._wake_event = threading.Event()

    def set_detector(self, detector, variant_method, threshold, overlay_alpha, confidence=0.5):
        """Update detector configuration."""
        self.detector = detector
        self.variant_method = variant_method
        self.threshold = threshold
        self.overlay_alpha = overlay_alpha
        self.confidence = confidence

    def set_overlay_alpha(self, overlay_alpha: float):
        """Update overlay alpha without resetting the detector configuration."""
        self.overlay_alpha = overlay_alpha

    def pause_processing(self):
        """Immediately pause inference processing."""
        self.processing_enabled = False
        with self.frame_lock:
            self.pending_frame = None  # Clear any pending work
        self._wake_event.set()

    def resume_processing(self):
        """Resume inference processing."""
        self.processing_enabled = True
        self._wake_event.set()

    def submit_frame(self, img_np):
        """Submit new frame for inference (always overwrites - keeps only newest)."""
        # Only accept frames if processing is enabled
        if not self.processing_enabled:
            return

        with self.frame_lock:
            self.pending_frame = img_np
        self._wake_event.set()

    def run(self):
        """Continuously process pending frames."""
        import time
        while self.running:
            self._wake_event.wait()
            self._wake_event.clear()

            if not self.running:
                break

            if not self.processing_enabled:
                continue

            # Get pending frame (thread-safe)
            with self.frame_lock:
                frame = self.pending_frame
                self.pending_frame = None  # Clear immediately

            if frame is not None and self.detector is not None and self.processing_enabled:
                try:
                    # Run inference with motion detection and timing
                    result = self.detector.process_frame(frame)

                    # Extract values from result dictionary
                    anomaly_map = result.get('anomaly_map', None)
                    max_score = result.get('max_score', 0.0)
                    inference_time_ms = result.get('total_inference_ms', 0.0)
                    index_search_ms = result.get('index_search_ms', 0.0)
                    detection_skipped = result.get('detection_skipped', False)
                    motion_active = result.get('motion_active', False)
                    motion_amount = result.get('motion_amount', 0.0)
                    current_fps = result.get('current_fps', 0.0)

                    # Handle motion detection case
                    if detection_skipped:
                        # Motion detected - skip anomaly detection, show original frame
                        final_img = frame
                        visualization_time_ms = 0.0
                        anomaly_map = None
                        max_score = 0.0
                    else:
                        # Validate
                        if anomaly_map is None or max_score is None or np.isnan(max_score):
                            final_img = frame
                            visualization_time_ms = 0.0
                        else:
                            # Create overlay (measure time)
                            t_viz_start = time.perf_counter()
                            final_img = self.variant_method(
                                frame,
                                anomaly_map,
                                threshold=self.threshold,
                                max_score=max_score,
                                alpha=self.overlay_alpha,
                                confidence=self.confidence
                            )
                            t_viz_end = time.perf_counter()
                            visualization_time_ms = (t_viz_end - t_viz_start) * 1000.0

                    # Emit result with timing info and motion status
                    self.inference_ready.emit(
                        anomaly_map,
                        final_img,
                        max_score,
                        inference_time_ms,
                        index_search_ms,
                        visualization_time_ms,
                        motion_active,
                        motion_amount,
                        current_fps
                    )

                except Exception as e:
                    print(f"[ERROR] Inference thread: {e}")
                    import traceback
                    traceback.print_exc()

    def stop(self):
        """Stop inference thread."""
        self.running = False
        self._wake_event.set()


class CameraThread(QThread):
    """Background thread for IDS camera frame acquisition."""

    frame_ready = Signal(object, object)  # image, buffer

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True

    def run(self):
        """Continuously fetch frames from IDS camera."""
        while self.running:
            try:
                image_view, buffer = self.camera.wait_for_image_view(100)

                if buffer.IsIncomplete():
                    self.camera.queue_buffer(buffer)
                    continue

                # Emit all frames (InferenceThread handles skipping)
                self.frame_ready.emit(image_view, buffer)

            except ids_peak.TimeoutException:
                continue
            except CommonException:
                continue
            except Exception as e:
                print(f"[ERROR] Camera thread: {e}")
                break

    def stop(self):
        """Stop camera thread."""
        self.running = False


class USBCameraThread(QThread):
    """Background thread for USB camera frame acquisition."""

    frame_ready = Signal(object, object)  # image, buffer

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True

    def run(self):
        """Continuously fetch frames from USB camera."""
        from usb_camera import TimeoutException

        while self.running:
            try:
                image_view, buffer = self.camera.wait_for_image_view(100)

                if buffer.IsIncomplete():
                    self.camera.queue_buffer(buffer)
                    continue

                # Emit all frames (InferenceThread handles skipping)
                self.frame_ready.emit(image_view, buffer)

            except TimeoutException:
                continue
            except Exception as e:
                print(f"[ERROR] USB Camera thread: {e}")
                break

    def stop(self):
        """Stop USB camera thread."""
        self.running = False
