"""Capture handler for image capture workflows."""
import cv2
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import QTimer
from gui.dialogs import ModelSelectionDialog
from gui.utils import image_to_numpy_rgb
from app_state import AppState


class CaptureHandler:
    """Handles image capture workflows (train, test)."""

    def __init__(self, host):
        self.host = host

    # ========== Capture Workflow ==========

    def start_capture_workflow(self):
        """Start image capture workflow - STARTS with STEP 1 intro (background)."""

        # START with STEP 1 intro (3s text overlay, then background capture)
        self.host.app_state = AppState.SHOW_STEP_1_INTRO
        self.host.captured_images = []
        self.host.capture_target = 5  # Always 5 background images
        self.host.capture_category = "background"
        self.host.update_instruction_ui()

    def on_action_button_press(self):
        """Unified handler for action button."""

        if self.host.app_state in [
            AppState.CAPTURE_BACKGROUND,
            AppState.CAPTURE_GOOD,  # Merged train+test good
            AppState.CAPTURE_TEST_DEFECT
        ]:
            self.capture_image()
        elif self.host.app_state == AppState.LIVE_DETECTION:
            self.host.detection_handler.toggle_detection()
        elif self.host.app_state == AppState.PROJECT_SELECTION:
            self.start_capture_workflow()

    def capture_image(self):
        """Capture current frame."""

        if self.host.last_frame is None:
            print("[DEBUG] No frame available, cannot capture")
            return

        # Get last displayed frame
        img_np = image_to_numpy_rgb(self.host.last_frame)

        # Determine category for saving
        if self.host.app_state == AppState.CAPTURE_GOOD:
            # CAPTURE_GOOD: split between train/good and test/good
            config = self.host.project_manager.current_config
            train_target = config.train_target if config else 16
            current_count = len(self.host.captured_images) + 1

            if current_count <= train_target:
                # First N images go to train/good
                category = "train/good"
                index = current_count
            else:
                # Remaining images go to test/good
                category = "test/good"
                index = current_count - train_target

            self.host.project_manager.save_image(img_np, category, index)
            print(f"[OK] Captured image for {category} ({index})")

            # Save first image as reference (first train/good image)
            if current_count == 1:
                self.host.project_manager.save_reference_image(img_np)
        else:
            # Other capture modes: use stored category
            index = len(self.host.captured_images) + 1
            self.host.project_manager.save_image(img_np, self.host.capture_category, index)
            print(f"[OK] Captured image {index} for {self.host.capture_category}")

            # Save first image as reference (for train/good only)
            if index == 1 and self.host.capture_category == "train/good":
                self.host.project_manager.save_reference_image(img_np)

        # Track captured count
        self.host.captured_images.append(len(self.host.captured_images) + 1)

        # Update progress
        self.host.progress_bar.setValue(len(self.host.captured_images))
        self.host.progress_label.setText(f"{len(self.host.captured_images)} / {self.host.capture_target}")

        # Update UNDO button visibility
        if len(self.host.captured_images) > 0:
            self.host.undo_button.setEnabled(True)
            self.host.undo_button.show()

        # Check if done with current phase
        if len(self.host.captured_images) >= self.host.capture_target:
            self.advance_capture_phase()

    def undo_last_capture(self):
        """Undo last captured image."""
        if not self.host.captured_images:
            return

        # Get last captured count
        last_count = self.host.captured_images[-1]

        # Remove from list
        self.host.captured_images.pop()

        # Determine category and index for deletion
        if self.host.app_state == AppState.CAPTURE_GOOD:
            # CAPTURE_GOOD: determine if last image was train/good or test/good
            config = self.host.project_manager.current_config
            train_target = config.train_target if config else 16

            if last_count <= train_target:
                # Was in train/good
                category = "train/good"
                index = last_count
            else:
                # Was in test/good
                category = "test/good"
                index = last_count - train_target

            # Delete file from disk
            try:
                img_path = self.host.project_manager.get_image_path(category, index)
                if img_path.exists():
                    img_path.unlink()
                    print(f"[OK] Deleted image {index} from {category}")
            except Exception as e:
                print(f"[ERROR] Failed to delete image: {e}")
        else:
            # Other capture modes: use stored category
            last_index = last_count

            # Delete file from disk
            try:
                img_path = self.host.project_manager.get_image_path(self.host.capture_category, last_index)
                if img_path.exists():
                    img_path.unlink()
                    print(f"[OK] Deleted image {last_index} from {self.host.capture_category}")
            except Exception as e:
                print(f"[ERROR] Failed to delete image: {e}")

        # Update progress
        self.host.progress_bar.setValue(len(self.host.captured_images))
        self.host.progress_label.setText(f"{len(self.host.captured_images)} / {self.host.capture_target}")

        # Hide UNDO button if no images
        if len(self.host.captured_images) == 0:
            self.host.undo_button.setEnabled(False)
            self.host.undo_button.hide()

        print(f"[OK] Undone capture #{last_count}")

    def advance_capture_phase(self):
        """Move to next capture phase with text overlay intros."""

        # Get targets from config
        config = self.host.project_manager.current_config
        train_target = config.train_target if config else 16
        test_good_target = config.test_good_target if config else 10
        test_defect_target = config.test_defect_target if config else 10

        if self.host.app_state == AppState.CAPTURE_BACKGROUND:
            # Move from background to STEP 2 intro (good images)
            self.host.app_state = AppState.SHOW_STEP_2_INTRO
            self.host.captured_images = []
            self.host.capture_target = train_target + test_good_target  # Combined good images
            self.host.capture_category = "good"  # Will be split internally
            self.host.update_instruction_ui()

        elif self.host.app_state == AppState.CAPTURE_GOOD:
            # Move from good to STEP 3 intro (defect images)
            self.host.app_state = AppState.SHOW_STEP_3_INTRO
            self.host.captured_images = []
            self.host.capture_target = test_defect_target
            self.host.capture_category = "test/defect"
            self.host.update_instruction_ui()

        elif self.host.app_state == AppState.CAPTURE_TEST_DEFECT:
            # Show model selection before training
            QMessageBox.information(self.host, "Bildaufnahme abgeschlossen",
                "Alle Bilder erfolgreich aufgenommen!\n\n"
                "Als naechstes waehlen Sie ein Modell und starten das Training.")
            QTimer.singleShot(100, self.show_model_selection_and_train)

    def show_model_selection_and_train(self):
        """Show model selection dialog before starting training."""

        # Show model selection dialog
        model_dialog = ModelSelectionDialog(self.host)

        if model_dialog.exec() == QDialog.DialogCode.Accepted:
            # Update project config with selected model and layers
            self.host.project_manager.current_config.model_name = model_dialog.selected_model
            self.host.project_manager.current_config.selected_layers = model_dialog.selected_layers
            self.host.project_manager.save_config()

            print(f"[OK] Selected model for training: {model_dialog.selected_model}")
            print(f"[OK] Selected layers: {model_dialog.selected_layers}")

            # Start training with selected model
            self.host.training_handler.start_training()
        else:
            QMessageBox.information(self.host, "Training abgebrochen",
                "Die Modellauswahl wurde abgebrochen.\n"
                "Sie koennen das Training spaeter ueber das Projekt-Menue starten.")
            # Return to project selection state
            self.host.app_state = AppState.PROJECT_SELECTION
            self.host.update_instruction_ui()
