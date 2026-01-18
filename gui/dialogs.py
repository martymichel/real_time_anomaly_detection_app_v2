"""
Dialog windows for user interaction.

Classes:
    - ProjectSelectionDialog: Dialog for project selection/creation
    - ProjectConfigDialog: Dialog for configuring new project image counts
    - ModelSelectionDialog: Dialog for selecting DINOv3 model before training
    - RetrainingConfigDialog: Dialog for retraining configuration
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QListWidget, QListWidgetItem, QComboBox, QCheckBox,
    QDialogButtonBox, QMessageBox, QWidget, QScrollArea
)
from PySide6.QtCore import Qt

# Backend modules
from project_manager import ProjectManager, ProjectConfig
from model_trainer import get_model_num_layers


class ProjectSelectionDialog(QDialog):
    """Dialog for project selection/creation."""

    def __init__(self, project_manager: ProjectManager, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.selected_project = None
        self.action = None  # 'load' or 'create'

        self.setWindowTitle("Project Selection")
        self.setModal(True)
        self.resize(500, 400)

        self.setup_ui()

    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout()

        # Existing projects section
        layout.addWidget(QLabel("<b>Select existing project:</b>"))

        self.project_list = QListWidget()
        self.project_list.itemDoubleClicked.connect(self.load_selected_project)

        # Populate project list
        projects = self.project_manager.list_projects()
        for proj in projects:
            item = QListWidgetItem(proj)
            self.project_list.addItem(item)

        if not projects:
            self.project_list.addItem(QListWidgetItem("No existing projects found."))
            self.project_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)

        layout.addWidget(self.project_list)

        # Button row for existing projects
        btn_row = QHBoxLayout()

        self.load_btn = QPushButton("LOAD SELECTED")
        self.load_btn.clicked.connect(self.load_selected_project)
        self.load_btn.setEnabled(len(projects) > 0)
        btn_row.addWidget(self.load_btn)

        self.delete_btn = QPushButton("DELETE SELECTED")
        self.delete_btn.clicked.connect(self.delete_selected_project)
        self.delete_btn.setEnabled(len(projects) > 0)
        btn_row.addWidget(self.delete_btn)

        self.retrain_btn = QPushButton("RETRAIN SELECTED")
        self.retrain_btn.clicked.connect(self.retrain_selected_project)
        self.retrain_btn.setEnabled(len(projects) > 0)
        btn_row.addWidget(self.retrain_btn)

        layout.addLayout(btn_row)

        # Separator
        separator = QLabel("─" * 60)
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(separator)

        # New project section
        layout.addWidget(QLabel("<b>Or create new project:</b>"))

        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("Enter project name")
        layout.addWidget(self.project_name_input)

        self.create_btn = QPushButton("CREATE NEW")
        self.create_btn.clicked.connect(self.create_new_project)
        layout.addWidget(self.create_btn)

        self.setLayout(layout)

    def load_selected_project(self):
        """Load selected project."""
        if not self.project_list.currentItem():
            return

        project_name = self.project_list.currentItem().text()
        if project_name == "No existing projects found.":
            return

        self.selected_project = project_name
        self.action = 'load'
        self.accept()

    def delete_selected_project(self):
        """Delete selected project."""
        if not self.project_list.currentItem():
            return

        project_name = self.project_list.currentItem().text()
        if project_name == "No existing projects found.":
            return

        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete project '{project_name}'?\n\n"
            "This will permanently delete all images, training data, and results.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.project_manager.delete_project(project_name)
                QMessageBox.information(
                    self,
                    "Project Deleted",
                    f"Project '{project_name}' was deleted successfully."
                )
                # Refresh list
                self.project_list.clear()
                projects = self.project_manager.list_projects()
                for proj in projects:
                    self.project_list.addItem(QListWidgetItem(proj))

                if not projects:
                    self.project_list.addItem(QListWidgetItem("No existing projects found."))
                    self.project_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
                    self.load_btn.setEnabled(False)
                    self.delete_btn.setEnabled(False)
                    self.retrain_btn.setEnabled(False)

            except Exception as e:
                QMessageBox.critical(self, "Deletion Error", str(e))

    def retrain_selected_project(self):
        """Retrain selected project."""
        if not self.project_list.currentItem():
            return

        project_name = self.project_list.currentItem().text()
        if project_name == "No existing projects found.":
            return

        self.selected_project = project_name
        self.action = 'retrain'
        self.accept()

    def create_new_project(self):
        """Create new project."""
        project_name = self.project_name_input.text().strip()

        if not project_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a project name.")
            return

        self.selected_project = project_name
        self.action = 'create'
        self.accept()


class ProjectConfigDialog(QDialog):
    """Dialog for configuring image counts for new project."""

    def __init__(self, project_name: str, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.train_target = 16
        self.test_good_target = 10
        self.test_defect_target = 10
        self.coreset_percentage = None  # None = disabled
        self.coreset_method = "random"  # Default coreset method
        self.image_size = 512  # Single training resolution

        self.setWindowTitle(f"Configure Project: {project_name}")
        self.setModal(True)
        self.resize(700, 780)  # 20% taller to show all options without scrolling

        self.setup_ui()

    def setup_ui(self):
        """Setup dialog UI."""
        main_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel(f"<b>Configure image counts for project:</b><br>{self.project_name}")
        title.setWordWrap(True)
        layout.addWidget(title)

        # Training images
        layout.addWidget(QLabel("\n<b>Training Images (good parts):</b>"))
        self.train_combo = QComboBox()
        self.train_combo.addItems(["1", "4", "8", "16", "32"])
        self.train_combo.setCurrentText("16")  # Default
        layout.addWidget(self.train_combo)

        # Test good images
        layout.addWidget(QLabel("\n<b>Test Images (good parts):</b>"))
        self.test_good_combo = QComboBox()
        self.test_good_combo.addItems(["5", "10", "20"])
        self.test_good_combo.setCurrentText("10")  # Default
        layout.addWidget(self.test_good_combo)

        # Test defect images
        layout.addWidget(QLabel("\n<b>Test Images (defect parts):</b>"))
        self.test_defect_combo = QComboBox()
        self.test_defect_combo.addItems(["5", "10", "20"])
        self.test_defect_combo.setCurrentText("10")  # Default
        layout.addWidget(self.test_defect_combo)

        # Coreset option
        layout.addWidget(QLabel("\n<b>Coreset Reduction (optional):</b>"))
        coreset_desc = QLabel(
            "Reduziert die Memorybank auf 1-10% für schnellere Inferenz.\n"
            "⚠ Training dauert etwas länger."
        )
        coreset_desc.setWordWrap(True)
        coreset_desc.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        layout.addWidget(coreset_desc)

        self.coreset_combo = QComboBox()
        self.coreset_combo.addItems(["Deaktiviert", "1%", "2%", "5%", "10%"])
        self.coreset_combo.setCurrentText("Deaktiviert")  # Default
        layout.addWidget(self.coreset_combo)

        # Coreset method selection
        layout.addWidget(QLabel("\n<b>Coreset Method:</b>"))
        coreset_method_desc = QLabel(
            "Algorithmus für Coreset-Auswahl:\n"
            "• Random: Schnellstes Training (~95% Qualität) ⚡ EMPFOHLEN\n"
            "• Stratified: Ausgewogen über Bilder (~97% Qualität)\n"
            "• FPS GPU: Beste Qualität (100%), langsameres Training\n"
            "• Importance: Diverse Features basierend auf Scores\n"
            "• Greedy: Legacy-Methode (langsam, nicht empfohlen)"
        )
        coreset_method_desc.setWordWrap(True)
        coreset_method_desc.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        layout.addWidget(coreset_method_desc)

        self.coreset_method_combo = QComboBox()
        self.coreset_method_combo.addItems(["random", "stratified", "fps_gpu", "importance", "greedy"])
        self.coreset_method_combo.setCurrentText("random")  # Default
        layout.addWidget(self.coreset_method_combo)

        # Training resolution (single resolution only)
        layout.addWidget(QLabel("\n<b>Trainingsauflösung:</b>"))
        resolution_desc = QLabel(
            "Die Memory Bank wird mit dieser Auflösung trainiert.\n"
            "Muss ein Vielfaches von 16 sein. Inferenz verwendet dieselbe Auflösung."
        )
        resolution_desc.setWordWrap(True)
        resolution_desc.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        layout.addWidget(resolution_desc)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["128", "256", "384", "512", "768", "1024"])
        self.resolution_combo.setCurrentText("512")
        layout.addWidget(self.resolution_combo)

        # Motion Detection Settings
        layout.addWidget(QLabel("\n<b>Motion Detection (optional):</b>"))
        motion_desc = QLabel(
            "Pausiert Anomaly Detection bei Bewegung im Bild.\n"
            "Verhindert False Positives durch vorübergehende Objekte."
        )
        motion_desc.setWordWrap(True)
        motion_desc.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        layout.addWidget(motion_desc)

        self.motion_enable_cb = QCheckBox("Motion Detection aktivieren")
        layout.addWidget(self.motion_enable_cb)

        # Motion settings (initially hidden)
        self.motion_settings_widget = QWidget()
        motion_settings_layout = QVBoxLayout()
        motion_settings_layout.setContentsMargins(20, 0, 0, 0)

        # Motion High Threshold
        motion_settings_layout.addWidget(QLabel("Motion Trigger Threshold:"))
        self.motion_high_combo = QComboBox()
        self.motion_high_combo.addItems(["1%", "2%", "3%", "5%", "7%", "10%"])
        self.motion_high_combo.setCurrentText("5%")
        motion_settings_layout.addWidget(self.motion_high_combo)

        # Motion Low Threshold
        motion_settings_layout.addWidget(QLabel("Motion Release Threshold:"))
        self.motion_low_combo = QComboBox()
        self.motion_low_combo.addItems(["0.1%", "0.5%", "1%", "2%", "3%"])
        self.motion_low_combo.setCurrentText("3%")
        motion_settings_layout.addWidget(self.motion_low_combo)

        # Stabilization Time
        motion_settings_layout.addWidget(QLabel("Stabilization Time (Sekunden):"))
        self.motion_stab_combo = QComboBox()
        self.motion_stab_combo.addItems(["0.5", "1.0", "2.0", "3.0", "5.0"])
        self.motion_stab_combo.setCurrentText("1.0")
        motion_settings_layout.addWidget(self.motion_stab_combo)

        self.motion_settings_widget.setLayout(motion_settings_layout)
        self.motion_settings_widget.hide()  # Hidden by default
        layout.addWidget(self.motion_settings_widget)

        # Connect checkbox to show/hide settings
        self.motion_enable_cb.toggled.connect(self.on_motion_enable_changed)

        # Buttons
        layout.addWidget(QLabel(""))  # Spacer
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept_config)
        button_box.rejected.connect(self.reject)
        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(button_box)
        self.setLayout(main_layout)

    def on_motion_enable_changed(self, checked: bool):
        """Show/hide motion settings based on checkbox."""
        if checked:
            self.motion_settings_widget.show()
        else:
            self.motion_settings_widget.hide()

    def accept_config(self):
        """Accept configuration."""
        self.train_target = int(self.train_combo.currentText())
        self.test_good_target = int(self.test_good_combo.currentText())
        self.test_defect_target = int(self.test_defect_combo.currentText())

        # Parse coreset percentage
        coreset_text = self.coreset_combo.currentText()
        if coreset_text == "Deaktiviert":
            self.coreset_percentage = None
        else:
            # Extract percentage value (e.g., "5%" -> 5.0)
            self.coreset_percentage = float(coreset_text.replace("%", ""))

        # Get selected coreset method
        self.coreset_method = self.coreset_method_combo.currentText()

        # Get training resolution
        self.image_size = int(self.resolution_combo.currentText())

        # Motion Detection parameters
        self.enable_motion_filter = self.motion_enable_cb.isChecked()
        if self.enable_motion_filter:
            # Parse motion thresholds (convert "5%" -> 0.05)
            self.motion_high_threshold = float(self.motion_high_combo.currentText().replace("%", "")) / 100.0
            self.motion_low_threshold = float(self.motion_low_combo.currentText().replace("%", "")) / 100.0
            self.motion_stabilization_time = float(self.motion_stab_combo.currentText())
        else:
            # Default values when disabled
            self.motion_high_threshold = 0.05
            self.motion_low_threshold = 0.01
            self.motion_stabilization_time = 2.0

        self.accept()


class ModelSelectionDialog(QDialog):
    """Dialog for selecting DINOv3 model before training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_model = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # Default
        self.selected_layers = [-1]  # Default: last layer only
        self.layer_checkboxes = []

        self.setWindowTitle("Modellauswahl")
        self.setModal(True)
        self.resize(650, 600)  # Wider for better readability

        self.setup_ui()

    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("<b>Wählen Sie ein DINOv3 Modell:</b>")
        layout.addWidget(title)

        # Description
        desc = QLabel(
            "Größere Modelle sind genauer, aber langsamer.\n"
            "Für Echtzeit-Anwendungen wird ViT-B/16 empfohlen."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #b0b0b0; font-size: 12px;")
        layout.addWidget(desc)

        layout.addWidget(QLabel(""))  # Spacer

        # Model selection
        self.model_combo = QComboBox()

        # Check which models are available locally
        models_dir = Path("models")
        available_models = []

        model_info = [
            ("facebook/dinov3-vits16-pretrain-lvd1689m", "DINOv3 ViT-S/16 (Small) - Schnellste", "vits16"),
            ("facebook/dinov3-vitb16-pretrain-lvd1689m", "DINOv3 ViT-B/16 (Base) - Empfohlen", "vitb16"),
            ("facebook/dinov3-vitl16-pretrain-lvd1689m", "DINOv3 ViT-L/16 (Large) - Höchste Genauigkeit", "vitl16"),
        ]

        for model_name, display_name, short_name in model_info:
            # Check if model exists locally
            model_path = models_dir / model_name.replace("/", "_")
            if model_path.exists():
                available_models.append((model_name, f"{display_name} ✓ lokal", short_name))
            else:
                available_models.append((model_name, f"{display_name} (Download)", short_name))

        for model_name, display_name, short_name in available_models:
            self.model_combo.addItem(display_name, model_name)

        # Set default to ViT-B/16 (index 1)
        self.model_combo.setCurrentIndex(1)

        # Connect model change to update layer selection
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        layout.addWidget(QLabel("<b>Modell:</b>"))
        layout.addWidget(self.model_combo)

        # Model specs
        layout.addWidget(QLabel(""))  # Spacer

        specs_text = QLabel(
            "<b>Modell-Spezifikationen:</b><br>"
            "• <b>ViT-S/16</b>: ~90 MB, sehr schnell, gute Genauigkeit<br>"
            "• <b>ViT-B/16</b>: ~330 MB, schnell, sehr gute Genauigkeit<br>"
            "• <b>ViT-L/16</b>: ~1.1 GB, langsam, beste Genauigkeit"
        )
        specs_text.setWordWrap(True)
        specs_text.setStyleSheet("font-size: 11px; color: #909090;")
        layout.addWidget(specs_text)

        # Layer selection
        layout.addWidget(QLabel(""))  # Spacer
        layout.addWidget(QLabel("<b>Layer-Auswahl für Feature-Extraktion:</b>"))

        layer_desc = QLabel(
            "Wählen Sie aus, welche Layer des Modells verwendet werden sollen.\n"
            "Der letzte Layer ist standardmäßig aktiv und kann nicht deaktiviert werden."
        )
        layer_desc.setWordWrap(True)
        layer_desc.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        layout.addWidget(layer_desc)

        # Container for layer checkboxes (will be populated dynamically)
        self.layer_checkbox_widget = QWidget()
        self.layer_checkbox_layout = QVBoxLayout()
        self.layer_checkbox_layout.setContentsMargins(20, 5, 0, 5)
        self.layer_checkbox_widget.setLayout(self.layer_checkbox_layout)
        layout.addWidget(self.layer_checkbox_widget)

        # Initialize layer checkboxes for current model
        self.update_layer_checkboxes()

        # Buttons
        layout.addWidget(QLabel(""))  # Spacer
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept_selection)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def on_model_changed(self):
        """Update layer checkboxes when model changes."""
        self.update_layer_checkboxes()

    def update_layer_checkboxes(self):
        """Update layer checkboxes based on selected model."""
        # Clear existing checkboxes
        for checkbox in self.layer_checkboxes:
            self.layer_checkbox_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self.layer_checkboxes.clear()

        # Get number of layers for selected model
        model_name = self.model_combo.currentData()
        num_layers = get_model_num_layers(model_name)

        # Create checkboxes for each layer (show every 3rd layer + last layer to avoid clutter)
        displayed_layers = []

        # Add intermediate layers (every 3rd)
        for i in range(0, num_layers, 3):
            if i > 0:  # Skip layer 0
                displayed_layers.append(i)

        # Always show last layer
        if num_layers - 1 not in displayed_layers:
            displayed_layers.append(num_layers - 1)

        for layer_idx in displayed_layers:
            checkbox = QCheckBox(f"Layer {layer_idx}")

            # Last layer is checked by default and disabled (cannot be unchecked)
            if layer_idx == num_layers - 1:
                checkbox.setChecked(True)
                checkbox.setEnabled(False)
                checkbox.setStyleSheet("font-weight: bold;")

            self.layer_checkbox_layout.addWidget(checkbox)
            self.layer_checkboxes.append(checkbox)

    def accept_selection(self):
        """Accept model selection and collect selected layers."""
        self.selected_model = self.model_combo.currentData()

        # Collect selected layers
        model_name = self.model_combo.currentData()
        num_layers = get_model_num_layers(model_name)
        selected = []

        # Map checkbox indices to actual layer indices
        displayed_layers = []
        for i in range(0, num_layers, 3):
            if i > 0:
                displayed_layers.append(i)
        if num_layers - 1 not in displayed_layers:
            displayed_layers.append(num_layers - 1)

        for idx, checkbox in enumerate(self.layer_checkboxes):
            if checkbox.isChecked():
                selected.append(displayed_layers[idx])

        # If no layers selected (shouldn't happen due to disabled last layer), default to last
        if not selected:
            self.selected_layers = [-1]
        else:
            # Convert to indices (-1 for last layer)
            self.selected_layers = []
            for layer_idx in selected:
                if layer_idx == num_layers - 1:
                    self.selected_layers.append(-1)  # Use -1 for last layer
                else:
                    self.selected_layers.append(layer_idx)

        self.accept()


class RetrainingConfigDialog(QDialog):
    """Dialog for retraining configuration."""

    def __init__(self, current_config: ProjectConfig, parent=None):
        super().__init__(parent)
        self.current_config = current_config

        # Retraining parameters (start with current values)
        self.model_name = current_config.model_name
        self.shots = current_config.shots
        self.knn_k = current_config.knn_k
        self.coreset_percentage = current_config.coreset_percentage
        self.coreset_method = current_config.coreset_method if hasattr(current_config, 'coreset_method') else "random"
        self.selected_layers = current_config.selected_layers if current_config.selected_layers else [-1]
        self.layer_checkboxes = []
        self.capture_new_images = False
        self.train_target = current_config.train_target
        self.test_good_target = current_config.test_good_target
        self.test_defect_target = current_config.test_defect_target
        self.image_size = current_config.image_size  # Single training resolution

        self.setWindowTitle(f"Retrain: {current_config.project_name}")
        self.setModal(True)
        self.resize(750, 850)  # Wider to show all options without scrolling

        self.setup_ui()

    def setup_ui(self):
        """Setup dialog UI."""
        main_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel(f"<b>Projekt neu trainieren:</b><br>{self.current_config.project_name}")
        title.setStyleSheet("font-size: 14px;")
        title.setWordWrap(True)
        layout.addWidget(title)

        # Current config info
        current_model_display = self.current_config.model_name.split('/')[-1] if '/' in self.current_config.model_name else self.current_config.model_name
        coreset_display = f"{self.current_config.coreset_percentage}%" if self.current_config.coreset_percentage else "Deaktiviert"

        info = QLabel(
            f"<b>Aktuelle Konfiguration:</b>\n"
            f"Modell: {current_model_display}\n"
            f"Shots: {self.current_config.shots} | KNN-K: {self.current_config.knn_k}\n"
            f"Coreset: {coreset_display}"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #b0b0b0; font-size: 11px; background-color: #2d2d2d; padding: 8px; border-radius: 4px;")
        layout.addWidget(info)

        layout.addWidget(QLabel(""))  # Spacer

        # Model selection
        layout.addWidget(QLabel("<b>Modell:</b>"))
        self.model_combo = QComboBox()

        models_dir = Path("models")
        model_info = [
            ("facebook/dinov3-vits16-pretrain-lvd1689m", "DINOv3 ViT-S/16 (Small)"),
            ("facebook/dinov3-vitb16-pretrain-lvd1689m", "DINOv3 ViT-B/16 (Base)"),
            ("facebook/dinov3-vitl16-pretrain-lvd1689m", "DINOv3 ViT-L/16 (Large)"),
        ]

        for model_name, display_name in model_info:
            model_path = models_dir / model_name.replace("/", "_")
            if model_path.exists():
                self.model_combo.addItem(f"{display_name} ✓ lokal", model_name)
            else:
                self.model_combo.addItem(f"{display_name} (Download)", model_name)

        # Set current model as default
        current_index = self.model_combo.findData(self.current_config.model_name)
        if current_index >= 0:
            self.model_combo.setCurrentIndex(current_index)

        # Connect model change to update layer selection
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        layout.addWidget(self.model_combo)

        # Shots
        layout.addWidget(QLabel("\n<b>Training Shots:</b>"))
        self.shots_combo = QComboBox()
        self.shots_combo.addItems(["1", "4", "8", "16", "32"])
        self.shots_combo.setCurrentText(str(self.current_config.shots))
        layout.addWidget(self.shots_combo)

        # KNN-K
        layout.addWidget(QLabel("\n<b>KNN-K (Nearest Neighbors):</b>"))
        self.knn_combo = QComboBox()
        self.knn_combo.addItems(["1", "3", "5", "7", "9"])
        self.knn_combo.setCurrentText(str(self.current_config.knn_k))
        layout.addWidget(self.knn_combo)

        # Coreset
        layout.addWidget(QLabel("\n<b>Coreset Reduction:</b>"))
        self.coreset_combo = QComboBox()
        self.coreset_combo.addItems(["Deaktiviert", "1%", "2%", "5%", "10%"])

        # Set current coreset value
        if self.current_config.coreset_percentage is None:
            self.coreset_combo.setCurrentText("Deaktiviert")
        else:
            self.coreset_combo.setCurrentText(f"{int(self.current_config.coreset_percentage)}%")

        layout.addWidget(self.coreset_combo)

        # Coreset method selection
        layout.addWidget(QLabel("\n<b>Coreset Method:</b>"))
        coreset_method_desc = QLabel(
            "Algorithmus für Coreset-Auswahl:\n"
            "• Random: Schnellstes Training (~95% Qualität) ⚡ EMPFOHLEN\n"
            "• Stratified: Ausgewogen über Bilder (~97% Qualität)\n"
            "• FPS GPU: Beste Qualität (100%), langsameres Training\n"
            "• Importance: Diverse Features basierend auf Scores\n"
            "• Greedy: Legacy-Methode (langsam, nicht empfohlen)"
        )
        coreset_method_desc.setWordWrap(True)
        coreset_method_desc.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        layout.addWidget(coreset_method_desc)

        self.coreset_method_combo = QComboBox()
        self.coreset_method_combo.addItems(["random", "stratified", "fps_gpu", "importance", "greedy"])
        # Set current coreset method
        current_method = self.current_config.coreset_method if hasattr(self.current_config, 'coreset_method') else "random"
        self.coreset_method_combo.setCurrentText(current_method)
        layout.addWidget(self.coreset_method_combo)

        # Training resolution (single resolution only)
        layout.addWidget(QLabel("\n<b>Trainingsauflösung:</b>"))
        resolution_desc = QLabel(
            "Die Memory Bank wird mit dieser Auflösung trainiert.\n"
            "Muss ein Vielfaches von 16 sein. Inferenz verwendet dieselbe Auflösung."
        )
        resolution_desc.setWordWrap(True)
        resolution_desc.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        layout.addWidget(resolution_desc)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["128", "256", "384", "512", "768", "1024"])
        self.resolution_combo.setCurrentText(str(self.current_config.image_size))
        layout.addWidget(self.resolution_combo)

        # Layer selection
        layout.addWidget(QLabel("\n<b>Layer-Auswahl für Feature-Extraktion:</b>"))

        layer_desc = QLabel(
            "Wählen Sie aus, welche Layer des Modells verwendet werden sollen.\n"
            "Der letzte Layer ist standardmäßig aktiv und kann nicht deaktiviert werden."
        )
        layer_desc.setWordWrap(True)
        layer_desc.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        layout.addWidget(layer_desc)

        # Container for layer checkboxes (will be populated dynamically)
        self.layer_checkbox_widget = QWidget()
        self.layer_checkbox_layout = QVBoxLayout()
        self.layer_checkbox_layout.setContentsMargins(20, 5, 0, 5)
        self.layer_checkbox_widget.setLayout(self.layer_checkbox_layout)
        layout.addWidget(self.layer_checkbox_widget)

        # Initialize layer checkboxes for current model
        self.update_layer_checkboxes()

        layout.addWidget(QLabel(""))  # Spacer

        # Checkbox: Capture new images
        self.capture_checkbox = QCheckBox("Neue Bilder erfassen (vor dem Training)")
        self.capture_checkbox.setStyleSheet("font-size: 13px; font-weight: bold;")
        self.capture_checkbox.toggled.connect(self.on_capture_checkbox_changed)
        layout.addWidget(self.capture_checkbox)

        # Image count configuration (hidden by default)
        self.image_config_widget = QWidget()
        image_config_layout = QVBoxLayout()
        image_config_layout.setContentsMargins(20, 0, 0, 0)

        image_config_layout.addWidget(QLabel("<b>Bildanzahl (neu erfassen):</b>"))

        # Train target
        image_config_layout.addWidget(QLabel("Training (gut):"))
        self.train_combo = QComboBox()
        self.train_combo.addItems(["1", "4", "8", "16", "32"])
        self.train_combo.setCurrentText(str(self.current_config.train_target))
        image_config_layout.addWidget(self.train_combo)

        # Test good target
        image_config_layout.addWidget(QLabel("Test (gut):"))
        self.test_good_combo = QComboBox()
        self.test_good_combo.addItems(["10", "20"])
        self.test_good_combo.setCurrentText(str(self.current_config.test_good_target))
        image_config_layout.addWidget(self.test_good_combo)

        # Test defect target
        image_config_layout.addWidget(QLabel("Test (defekt):"))
        self.test_defect_combo = QComboBox()
        self.test_defect_combo.addItems(["10", "20"])
        self.test_defect_combo.setCurrentText(str(self.current_config.test_defect_target))
        image_config_layout.addWidget(self.test_defect_combo)

        self.image_config_widget.setLayout(image_config_layout)
        self.image_config_widget.hide()  # Hidden by default
        layout.addWidget(self.image_config_widget)

        # Buttons
        layout.addWidget(QLabel(""))  # Spacer
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept_config)
        button_box.rejected.connect(self.reject)
        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(button_box)
        self.setLayout(main_layout)

    def on_capture_checkbox_changed(self, checked: bool):
        """Show/hide image count configuration based on checkbox."""
        if checked:
            self.image_config_widget.show()
        else:
            self.image_config_widget.hide()

    def on_model_changed(self):
        """Update layer checkboxes when model changes."""
        self.update_layer_checkboxes()

    def update_layer_checkboxes(self):
        """Update layer checkboxes based on selected model."""
        # Clear existing checkboxes
        for checkbox in self.layer_checkboxes:
            self.layer_checkbox_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self.layer_checkboxes.clear()

        # Get number of layers for selected model
        model_name = self.model_combo.currentData()
        num_layers = get_model_num_layers(model_name)

        # Create checkboxes for each layer (show every 3rd layer + last layer to avoid clutter)
        displayed_layers = []

        # Add intermediate layers (every 3rd)
        for i in range(0, num_layers, 3):
            if i > 0:  # Skip layer 0
                displayed_layers.append(i)

        # Always show last layer
        if num_layers - 1 not in displayed_layers:
            displayed_layers.append(num_layers - 1)

        for layer_idx in displayed_layers:
            checkbox = QCheckBox(f"Layer {layer_idx}")

            # Check if this layer was previously selected
            is_selected = False
            if layer_idx == num_layers - 1 and -1 in self.selected_layers:
                is_selected = True
            elif layer_idx in self.selected_layers:
                is_selected = True

            checkbox.setChecked(is_selected)

            # Last layer is always checked and disabled (cannot be unchecked)
            if layer_idx == num_layers - 1:
                checkbox.setChecked(True)
                checkbox.setEnabled(False)
                checkbox.setStyleSheet("font-weight: bold;")

            self.layer_checkbox_layout.addWidget(checkbox)
            self.layer_checkboxes.append(checkbox)

    def accept_config(self):
        """Accept configuration."""
        self.model_name = self.model_combo.currentData()
        self.shots = int(self.shots_combo.currentText())
        self.knn_k = int(self.knn_combo.currentText())

        # Parse coreset
        coreset_text = self.coreset_combo.currentText()
        if coreset_text == "Deaktiviert":
            self.coreset_percentage = None
        else:
            self.coreset_percentage = float(coreset_text.replace("%", ""))

        # Get selected coreset method
        self.coreset_method = self.coreset_method_combo.currentText()

        # Get training resolution
        self.image_size = int(self.resolution_combo.currentText())

        # Collect selected layers
        model_name = self.model_combo.currentData()
        num_layers = get_model_num_layers(model_name)
        selected = []

        # Map checkbox indices to actual layer indices
        displayed_layers = []
        for i in range(0, num_layers, 3):
            if i > 0:
                displayed_layers.append(i)
        if num_layers - 1 not in displayed_layers:
            displayed_layers.append(num_layers - 1)

        for idx, checkbox in enumerate(self.layer_checkboxes):
            if checkbox.isChecked():
                selected.append(displayed_layers[idx])

        # If no layers selected (shouldn't happen), default to last
        if not selected:
            self.selected_layers = [-1]
        else:
            # Convert to indices (-1 for last layer)
            self.selected_layers = []
            for layer_idx in selected:
                if layer_idx == num_layers - 1:
                    self.selected_layers.append(-1)  # Use -1 for last layer
                else:
                    self.selected_layers.append(layer_idx)

        self.capture_new_images = self.capture_checkbox.isChecked()

        if self.capture_new_images:
            self.train_target = int(self.train_combo.currentText())
            self.test_good_target = int(self.test_good_combo.currentText())
            self.test_defect_target = int(self.test_defect_combo.currentText())

        self.accept()


class CameraTypeSelectionDialog(QDialog):
    """Dialog for selecting camera type (IDS Peak or USB)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_type = None  # Will be "ids" or "usb"

        self.setWindowTitle("Kamera-Auswahl")
        self.setModal(True)
        self.resize(450, 250)

        self.setup_ui()

    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("<b>Welche Kamera möchten Sie verwenden?</b>")
        title.setStyleSheet("font-size: 16px;")
        layout.addWidget(title)

        layout.addWidget(QLabel(""))  # Spacer

        # Description
        desc = QLabel(
            "Wählen Sie den Kamera-Typ für diese Session.\n\n"
            "• <b>IDS Peak</b>: Industriekamera mit IDS Peak SDK\n"
            "• <b>USB Kamera</b>: Standard USB-Kamera (OpenCV)"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #e0e0e0;")
        layout.addWidget(desc)

        layout.addWidget(QLabel(""))  # Spacer

        # IDS Peak button
        self.ids_btn = QPushButton("IDS PEAK")
        self.ids_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5a9ff2;
            }
        """)
        self.ids_btn.clicked.connect(self.select_ids)
        layout.addWidget(self.ids_btn)

        # USB button
        self.usb_btn = QPushButton("USB KAMERA")
        self.usb_btn.setStyleSheet("""
            QPushButton {
                background-color: #50a050;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #60b060;
            }
        """)
        self.usb_btn.clicked.connect(self.select_usb)
        layout.addWidget(self.usb_btn)

        self.setLayout(layout)

    def select_ids(self):
        """Select IDS Peak camera."""
        self.camera_type = "ids"
        self.accept()

    def select_usb(self):
        """Select USB camera."""
        self.camera_type = "usb"
        self.accept()
