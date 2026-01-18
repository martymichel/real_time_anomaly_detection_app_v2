"""Training handler for model training workflow."""
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Slot
from gui.threads import TrainingThread
from app_state import AppState

class TrainingHandler:
    """Handles model training workflow."""
    def __init__(self, host):
        self.host = host
    
    def start_training(self):
        """Start model training in background thread."""
        self.host.app_state = AppState.TRAINING
        self.host.update_instruction_ui()
        
        config = self.host.project_manager.current_config
        self.host.training_thread = TrainingThread(
            self.host.project_manager, config.model_name, config.shots,
            config.batch_size, config.knn_k, config.metric
        )
        self.host.training_thread.progress_changed.connect(self.on_training_progress)
        self.host.training_thread.training_complete.connect(self.on_training_complete)
        self.host.training_thread.training_error.connect(self.on_training_error)
        self.host.training_thread.start()
    
    @Slot(str, float)
    def on_training_progress(self, message: str, value: float):
        self.host.progress_label.setText(message)
        self.host.progress_bar.setValue(int(value * 100))
    
    @Slot(object)
    def on_training_complete(self, memory_bank):
        self.host.progress_label.setText("Training complete!")
        self.host.progress_bar.setValue(100)
        summary = self.host.project_manager.get_project_summary()
        QMessageBox.information(self.host, "Training Complete!",
            f"Memory bank trained successfully!\n\n"
            f"Threshold: {summary['threshold']:.4f}\n"
            f"Accuracy: {summary['validation']['accuracy']:.2%}\n"
            f"F1-Score: {summary['validation']['f1_score']:.2%}\n\n"
            "Ready for live anomaly detection!")
        self.host.detection_handler.init_live_detection(memory_bank)
    
    @Slot(str)
    def on_training_error(self, error_msg: str):
        QMessageBox.critical(self.host, "Training Error", error_msg)
        self.host.app_state = AppState.PROJECT_SELECTION
        self.host.update_instruction_ui()
