"""UI builder for main window setup and styling."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QSlider, QCheckBox
)
from PySide6.QtCore import Qt, QTimer
from gui.widgets import MarkedSlider


class UIBuilder:
    """
    Builds and styles the main window UI.

    Handles:
        - UI layout creation
        - Dark theme styling
        - Timer setup
    """

    @staticmethod
    def build_main_ui(host):
        """
        Build main window UI layout.

        Args:
            host: Main window instance (AnomalyDetectionAppQt)
        """
        # Central widget
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        host.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        central_widget.setLayout(main_layout)

        # Top bar
        top_bar = QHBoxLayout()
        top_bar.setSpacing(20)

        host.project_button = QPushButton("PROJEKT")
        host.project_button.setFixedSize(200, 48)
        host.project_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevent spacebar from triggering this button
        host.project_button.clicked.connect(host.project_handler.show_project_selection_dialog)
        top_bar.addWidget(host.project_button)

        # Motion Filter Toggle Button (oben links)
        host.motion_filter_button = QPushButton("MOTION-FILTER: ON")
        host.motion_filter_button.setFixedSize(220, 48)
        host.motion_filter_button.setCheckable(True)
        host.motion_filter_button.setChecked(True)  # Default: ON (Motion Filter enabled)
        host.motion_filter_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevent spacebar from triggering this button
        host.motion_filter_button.clicked.connect(host.detection_handler.toggle_motion_filter)
        host.motion_filter_button.setStyleSheet("""
            QPushButton {
                background-color: #2d6d2d;
                border: 1px solid #408040;
                color: #90ff90;
            }
            QPushButton:hover {
                background-color: #3d7d3d;
            }
            QPushButton:checked {
                background-color: #6d2d2d;
                border: 1px solid #804040;
                color: #ff9090;
            }
            QPushButton:checked:hover {
                background-color: #7d3d3d;
            }
        """)
        host.motion_filter_button.hide()  # Hidden until LIVE_DETECTION mode
        top_bar.addWidget(host.motion_filter_button)

        # Visualization Mode Toggle Button
        host.viz_mode_button = QPushButton("VIZ: CLASSIC")
        host.viz_mode_button.setFixedSize(160, 48)
        host.viz_mode_button.setCheckable(True)
        host.viz_mode_button.setChecked(False)  # Default: CLASSIC (unchecked)
        host.viz_mode_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        host.viz_mode_button.clicked.connect(host.detection_handler.toggle_visualization_mode)
        host.viz_mode_button.setStyleSheet("""
            QPushButton {
                background-color: #2d4d6d;
                border: 1px solid #406080;
                color: #90c0ff;
            }
            QPushButton:hover {
                background-color: #3d5d7d;
            }
            QPushButton:checked {
                background-color: #6d4d2d;
                border: 1px solid #806040;
                color: #ffc090;
            }
            QPushButton:checked:hover {
                background-color: #7d5d3d;
            }
        """)
        host.viz_mode_button.hide()  # Hidden until LIVE_DETECTION mode
        top_bar.addWidget(host.viz_mode_button)

        top_bar.addStretch()

        host.status_label = QLabel("Kamera: Initialisierung...")
        top_bar.addWidget(host.status_label)

        host.fps_label = QLabel("FPS: 0.0")
        host.fps_label.setFixedWidth(180)
        host.fps_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        top_bar.addWidget(host.fps_label)

        main_layout.addLayout(top_bar)

        # Image display
        host.image_widget = QLabel()
        host.image_widget.setObjectName("image_widget")
        host.image_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        host.image_widget.setMinimumSize(640, 480)
        main_layout.addWidget(host.image_widget, stretch=1)

        # Toggle button for controls panel
        toggle_controls_layout = QHBoxLayout()
        toggle_controls_layout.setContentsMargins(0, 5, 0, 0)
        host.toggle_controls_button = QPushButton("▼ Controls ausblenden")
        host.toggle_controls_button.setFixedHeight(24)
        host.toggle_controls_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevent spacebar from triggering this button
        host.toggle_controls_button.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px;
                color: #b0b0b0;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
        """)
        host.toggle_controls_button.clicked.connect(host.toggle_controls_panel)
        toggle_controls_layout.addStretch()
        toggle_controls_layout.addWidget(host.toggle_controls_button)
        toggle_controls_layout.addStretch()
        main_layout.addLayout(toggle_controls_layout)

        # Footer bar (two-column layout)
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(20)
        footer_layout.setContentsMargins(0, 10, 0, 0)

        # LEFT COLUMN: Controls
        left_column = QVBoxLayout()
        left_column.setSpacing(8)

        # Instruction label
        host.instruction_label = QLabel("Warte auf Projektauswahl...")
        host.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        host.instruction_label.setWordWrap(True)
        host.instruction_label.setStyleSheet("font-size: 20px; color: #e0e0e0;")
        host.instruction_label.setMinimumHeight(60)
        left_column.addWidget(host.instruction_label)

        # Progress bar
        host.progress_bar = QProgressBar()
        host.progress_bar.setFixedHeight(24)
        host.progress_bar.setMaximum(100)
        left_column.addWidget(host.progress_bar)

        # Confidence slider
        confidence_row = QHBoxLayout()
        confidence_row.setSpacing(8)

        host.confidence_label = QLabel("Kontur-Confidence")
        host.confidence_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        host.confidence_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        host.confidence_label.hide()
        confidence_row.addWidget(host.confidence_label)

        host.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        host.confidence_slider.setMinimum(0)
        host.confidence_slider.setMaximum(1000)
        host.confidence_slider.setValue(50)  # 0.05 default for minimal noise
        host.confidence_slider.setFixedHeight(24)
        host.confidence_slider.valueChanged.connect(host.detection_handler.on_confidence_change)
        host.confidence_slider.hide()
        confidence_row.addWidget(host.confidence_slider, stretch=1)

        host.confidence_value_label = QLabel("0.05")
        host.confidence_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        host.confidence_value_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        host.confidence_value_label.setFixedWidth(60)
        host.confidence_value_label.hide()
        confidence_row.addWidget(host.confidence_value_label)

        left_column.addLayout(confidence_row)
        left_column.addSpacing(8)

        # Threshold slider (with marker for original trained threshold)
        threshold_row = QHBoxLayout()
        threshold_row.setSpacing(8)

        host.threshold_label = QLabel("Schwelle")
        host.threshold_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        host.threshold_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        host.threshold_label.hide()
        threshold_row.addWidget(host.threshold_label)

        host.threshold_slider = MarkedSlider(Qt.Orientation.Horizontal)
        host.threshold_slider.setMinimum(0)
        host.threshold_slider.setMaximum(1000)
        host.threshold_slider.setValue(500)
        host.threshold_slider.setFixedHeight(24)
        host.threshold_slider.valueChanged.connect(host.detection_handler.on_threshold_change)
        host.threshold_slider.hide()
        threshold_row.addWidget(host.threshold_slider, stretch=1)

        host.threshold_value_label = QLabel("0.500")
        host.threshold_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        host.threshold_value_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        host.threshold_value_label.setFixedWidth(60)
        host.threshold_value_label.hide()
        threshold_row.addWidget(host.threshold_value_label)

        left_column.addLayout(threshold_row)
        left_column.addSpacing(8)

        # Action buttons
        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        host.action_button = QPushButton("WARTEN")
        host.action_button.setFixedSize(180, 48)
        host.action_button.setEnabled(False)
        host.action_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevent spacebar from triggering this button
        host.action_button.clicked.connect(host.capture_handler.on_action_button_press)
        button_row.addWidget(host.action_button)

        host.undo_button = QPushButton("RUECKGAENGIG")
        host.undo_button.setFixedSize(140, 48)
        host.undo_button.setEnabled(False)
        host.undo_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevent spacebar from triggering this button
        host.undo_button.hide()
        host.undo_button.clicked.connect(host.capture_handler.undo_last_capture)
        button_row.addWidget(host.undo_button)

        left_column.addLayout(button_row)

        footer_layout.addLayout(left_column, stretch=1)

        # RIGHT COLUMN: Status information
        right_column = QVBoxLayout()
        right_column.setSpacing(4)

        # Progress label
        host.progress_label = QLabel("")
        host.progress_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        host.progress_label.setWordWrap(True)
        host.progress_label.setStyleSheet("font-size: 12px; color: #b0b0b0;")
        right_column.addWidget(host.progress_label)

        # Status text area
        host.status_text = QLabel("")
        host.status_text.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        host.status_text.setWordWrap(True)
        host.status_text.setStyleSheet("font-size: 14px; color: #b0b0b0;")
        host.status_text.setMinimumHeight(80)
        right_column.addWidget(host.status_text)

        right_column.addStretch()

        footer_layout.addLayout(right_column, stretch=1)

        # Wrap footer in widget for collapsing
        host.controls_panel = QWidget()
        host.controls_panel.setLayout(footer_layout)
        main_layout.addWidget(host.controls_panel)

    @staticmethod
    def apply_dark_theme(host):
        """Apply dark theme stylesheet to main window."""
        host.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-size: 14px;
            }
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px;
                color: #e0e0e0;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border: 1px solid #505050;
            }
            QPushButton:pressed {
                background-color: #4d4d4d;
            }
            QPushButton:disabled {
                background-color: #252525;
                color: #606060;
            }
            QLabel {
                background-color: transparent;
            }
            QProgressBar {
                border: 1px solid #404040;
                border-radius: 2px;
                background-color: #252525;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #008a96;
            }
            QSlider::groove:horizontal {
                border: 1px solid #404040;
                height: 8px;
                background: #252525;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #008a96;
                border: 1px solid #006070;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px;
                color: #e0e0e0;
            }
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #e0e0e0;
            }
            QListWidget::item {
                padding: 8px;
            }
            QListWidget::item:selected {
                background-color: #008a96;
            }
            QDialog {
                background-color: #1e1e1e;
            }
        """)

    @staticmethod
    def setup_timers(host):
        """Setup Qt timers for camera and FPS updates."""
        # Camera update timer (30 FPS)
        host.camera_timer = QTimer()
        host.camera_timer.timeout.connect(host.camera_handler.update_image)
        host.camera_timer.start(33)  # ~30 FPS

        # FPS counter timer (1 Hz)
        host.fps_timer = QTimer()
        host.fps_timer.timeout.connect(host.camera_handler.update_fps)
        host.fps_timer.start(1000)  # 1 second
