"""Project handler for project management workflow."""
from PySide6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import QTimer
from gui.dialogs import ProjectSelectionDialog, ProjectConfigDialog, ModelSelectionDialog, RetrainingConfigDialog
from project_manager import ProjectConfig
from app_state import AppState

class ProjectHandler:
    """Handles project selection, creation, loading, and retraining."""
    def __init__(self, host):
        self.host = host

    def show_project_selection_dialog(self):
        """Show dialog to select or create project."""
        dialog = ProjectSelectionDialog(self.host.project_manager, self.host)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.action == 'load':
                self.load_project(dialog.selected_project)
            elif dialog.action == 'create':
                self.create_new_project(dialog.selected_project)
            elif dialog.action == 'retrain':
                self.retrain_project(dialog.selected_project)

        self.host.showMaximized()

    def load_project(self, project_name: str):
        """Load existing project."""
        if self.host.project_manager.current_project == project_name:
            QMessageBox.information(self.host, "Project Already Loaded",
                f"Project '{project_name}' is already the active project.")
            return

        try:
            self.host.project_manager.load_project(project_name)

            memory_bank = self.host.project_manager.load_memory_bank()
            has_images = self.host.project_manager.has_training_images()

            if memory_bank is not None and self.host.project_manager.is_ready_for_inference():
                # Project is trained and ready - go directly to live detection!
                # No dialog needed, user can retrain later if needed via menu
                print(f"[INFO] Project '{project_name}' is ready for inference - starting live detection", flush=True)
                self.host.detection_handler.init_live_detection(memory_bank)
            else:
                if has_images:
                    self.show_resume_training_dialog(project_name)
                else:
                    QMessageBox.information(self.host, "Project Incomplete",
                        f"Project '{project_name}' exists but training is incomplete.\n"
                        "Please complete the data collection and training workflow.")
                    self.host.capture_handler.start_capture_workflow()
        except Exception as e:
            QMessageBox.critical(self.host, "Load Error", str(e))

    def create_new_project(self, project_name: str):
        """Create new project."""
        try:
            config_dialog = ProjectConfigDialog(project_name, self.host)
            if config_dialog.exec() != QDialog.DialogCode.Accepted:
                return

            config = ProjectConfig(
                project_name=project_name,
                model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
                shots=config_dialog.train_target,
                knn_k=5, batch_size=4,
                train_target=config_dialog.train_target,
                test_good_target=config_dialog.test_good_target,
                test_defect_target=config_dialog.test_defect_target,
                coreset_percentage=config_dialog.coreset_percentage,
                coreset_method=config_dialog.coreset_method,
                image_size=config_dialog.image_size,
                # Motion Detection Parameters
                enable_motion_filter=config_dialog.enable_motion_filter,
                motion_high_threshold=config_dialog.motion_high_threshold,
                motion_low_threshold=config_dialog.motion_low_threshold,
                motion_stabilization_time=config_dialog.motion_stabilization_time
            )

            self.host.project_manager.create_project(config)
            QMessageBox.information(self.host, "Projekt erstellt",
                f"Projekt '{project_name}' erfolgreich erstellt!\n\n"
                f"Bildanzahl:\n"
                f"  Training: {config.train_target} Bilder\n"
                f"  Test (gut): {config.test_good_target} Bilder\n"
                f"  Test (defekt): {config.test_defect_target} Bilder\n\n"
                "Naechster Schritt: Bildaufnahme-Workflow starten.")

            QTimer.singleShot(100, self.host.capture_handler.start_capture_workflow)
        except Exception as e:
            QMessageBox.critical(self.host, "Creation Error", str(e))

    def retrain_project(self, project_name: str):
        """Retrain existing project with new configuration."""
        try:
            self.host.project_manager.load_project(project_name)

            retrain_dialog = RetrainingConfigDialog(self.host.project_manager.current_config, self.host)
            if retrain_dialog.exec() != QDialog.DialogCode.Accepted:
                return

            config = self.host.project_manager.current_config
            config.model_name = retrain_dialog.model_name
            config.shots = retrain_dialog.shots
            config.knn_k = retrain_dialog.knn_k
            config.coreset_percentage = retrain_dialog.coreset_percentage
            config.coreset_method = retrain_dialog.coreset_method
            config.selected_layers = retrain_dialog.selected_layers
            config.image_size = retrain_dialog.image_size

            if retrain_dialog.capture_new_images:
                config.train_target = retrain_dialog.train_target
                config.test_good_target = retrain_dialog.test_good_target
                config.test_defect_target = retrain_dialog.test_defect_target
                self.host.project_manager.save_config()
                QTimer.singleShot(100, self.host.capture_handler.start_capture_workflow)
            else:
                self.host.project_manager.save_config()
                QTimer.singleShot(100, self.host.training_handler.start_training)
        except Exception as e:
            QMessageBox.critical(self.host, "Retrain Error", str(e))

    def show_trained_project_dialog(self, project_name: str, memory_bank):
        """Show dialog for already trained project."""
        train_path = self.host.project_manager.get_train_images_path() / "good"
        test_good_path, test_defect_path = self.host.project_manager.get_test_images_paths()

        train_count = len(list(train_path.glob("*.png"))) if train_path.exists() else 0
        test_good_count = len(list(test_good_path.glob("*.png"))) if test_good_path.exists() else 0
        test_defect_count = len(list(test_defect_path.glob("*.png"))) if test_defect_path.exists() else 0

        current_model = self.host.project_manager.current_config.model_name
        model_display = current_model.split('/')[-1] if '/' in current_model else current_model

        dialog = QDialog(self.host)
        dialog.setWindowTitle("Projekt geladen")
        dialog.setModal(True)
        dialog.resize(500, 350)

        layout = QVBoxLayout()
        title = QLabel(f"<b>Projekt '{project_name}' ist bereit!</b>")
        title.setStyleSheet("font-size: 16px;")
        layout.addWidget(title)

        info = QLabel(
            f"Das Projekt wurde bereits trainiert:\n\n"
            f"<b>Modell:</b> {model_display}\n"
            f"<b>Threshold:</b> {self.host.project_manager.current_config.threshold:.4f}\n\n"
            f"<b>Vorhandene Bilder:</b>\n"
            f"  {train_count} Trainingsbilder (gut)\n"
            f"  {test_good_count} Test-Bilder (gut)\n"
            f"  {test_defect_count} Test-Bilder (defekt)\n\n"
            f"Was moechten Sie tun?")
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_layout = QVBoxLayout()
        live_btn = QPushButton("LIVE DETECTION STARTEN")
        live_btn.setStyleSheet("font-size: 14px; padding: 12px; background-color: #008a96;")
        live_btn.clicked.connect(lambda: self._start_live_detection(dialog, memory_bank))
        btn_layout.addWidget(live_btn)

        retrain_btn = QPushButton("NEU TRAINIEREN (anderes Modell waehlen)")
        retrain_btn.clicked.connect(lambda: (dialog.accept(), self.start_retrain_workflow()))
        btn_layout.addWidget(retrain_btn)

        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        dialog.exec()

    def show_resume_training_dialog(self, project_name: str):
        """Show dialog to resume training."""
        train_path = self.host.project_manager.get_train_images_path() / "good"
        test_good_path, test_defect_path = self.host.project_manager.get_test_images_paths()

        train_count = len(list(train_path.glob("*.png"))) if train_path.exists() else 0
        test_good_count = len(list(test_good_path.glob("*.png"))) if test_good_path.exists() else 0
        test_defect_count = len(list(test_defect_path.glob("*.png"))) if test_defect_path.exists() else 0

        reply = QMessageBox.question(self.host, "Training starten",
            f"Projekt '{project_name}' hat bereits Bilder:\n\n"
            f"  {train_count} Trainingsbilder (gut)\n"
            f"  {test_good_count} Test-Bilder (gut)\n"
            f"  {test_defect_count} Test-Bilder (defekt)\n\n"
            "Moechten Sie das Training mit diesen Bildern durchfuehren?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            model_dialog = ModelSelectionDialog(self.host)
            if model_dialog.exec() == QDialog.DialogCode.Accepted:
                self.host.project_manager.current_config.model_name = model_dialog.selected_model
                self.host.project_manager.current_config.selected_layers = model_dialog.selected_layers
                self.host.project_manager.save_config()
                self.host.training_handler.start_training()

    def start_retrain_workflow(self):
        """Start retraining workflow with model selection."""
        model_dialog = ModelSelectionDialog(self.host)
        if model_dialog.exec() == QDialog.DialogCode.Accepted:
            self.host.project_manager.current_config.model_name = model_dialog.selected_model
            self.host.project_manager.current_config.selected_layers = model_dialog.selected_layers
            self.host.project_manager.save_config()
            self.host.training_handler.start_training()
        else:
            self.host.app_state = AppState.PROJECT_SELECTION
            self.host.update_instruction_ui()
