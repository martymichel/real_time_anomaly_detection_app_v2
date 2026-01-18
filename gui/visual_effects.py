"""
Visual effects mixin for GUI animations and styling.

Provides:
    - Background color animations
    - Overlay alpha animations
    - Theme color animations
    - Border color management
"""

from PySide6.QtCore import QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QColor


class VisualEffectsMixin:
    """
    Mixin class for visual effects and animations.

    Provides smooth animations for:
    - Background colors
    - Overlay alpha (anomaly map fade in/out)
    - Theme colors (all UI elements)
    - Stream border colors

    Requirements:
        - Host class must inherit from QWidget or QMainWindow
        - Host class must have self.image_widget and self.centralWidget()
    """

    def init_visual_effects(self):
        """Initialize visual effects state (call from host __init__)."""
        # Background color animation
        self._bg_color = QColor("#1e1e1e")  # Default dark background
        self._bg_animation = None

        # Overlay alpha animation
        self._overlay_alpha = 0.0  # Start hidden
        self._overlay_alpha_animation = None
        self._target_overlay_alpha = 0.5  # Target alpha when anomaly detected

        # Theme color animation
        self._current_theme_color = QColor("#1e1e1e")
        self._theme_animation = None

    # ========== Background Color Animation ==========

    def set_background_color(self, color: str):
        """
        Set background color instantly (no animation).

        Args:
            color: Color as hex string (e.g., "#1e1e1e")
        """
        self.centralWidget().setStyleSheet(f"QWidget#centralWidget {{ background-color: {color}; }}")

    def animate_background_color(self, target_color: str, duration: int = 500):
        """
        Animate background color transition smoothly.

        Args:
            target_color: Target color as hex string (e.g., "#4d1010")
            duration: Animation duration in milliseconds (default: 500ms)
        """
        # Stop existing animation if running
        if self._bg_animation is not None:
            self._bg_animation.stop()

        # Create color animation
        self._bg_animation = QPropertyAnimation(self, b"bg_color")
        self._bg_animation.setDuration(duration)
        self._bg_animation.setStartValue(self._bg_color)
        self._bg_animation.setEndValue(QColor(target_color))
        self._bg_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._bg_animation.start()

    def get_bg_color(self) -> QColor:
        """Get current background color (for Qt property system)."""
        return self._bg_color

    def set_bg_color(self, color: QColor):
        """Set current background color and update stylesheet (for Qt property system)."""
        self._bg_color = color
        rgb = f"rgb({color.red()}, {color.green()}, {color.blue()})"
        self.centralWidget().setStyleSheet(f"QWidget#centralWidget {{ background-color: {rgb}; }}")

    # Qt property for color animation
    bg_color = Property(QColor, get_bg_color, set_bg_color)

    # ========== Overlay Alpha Animation ==========

    def animate_overlay_alpha(self, target_alpha: float, duration: int = 400):
        """
        Animate overlay alpha transition smoothly.

        Args:
            target_alpha: Target alpha value (0.0 = hidden, 0.5 = half visible, etc.)
            duration: Animation duration in milliseconds (default: 400ms)
        """
        # Stop existing animation if running
        if self._overlay_alpha_animation is not None:
            self._overlay_alpha_animation.stop()

        # Create alpha animation
        self._overlay_alpha_animation = QPropertyAnimation(self, b"overlay_alpha")
        self._overlay_alpha_animation.setDuration(duration)
        self._overlay_alpha_animation.setStartValue(self._overlay_alpha)
        self._overlay_alpha_animation.setEndValue(target_alpha)
        self._overlay_alpha_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._overlay_alpha_animation.start()

    def get_overlay_alpha(self) -> float:
        """Get current overlay alpha (for Qt property system)."""
        return self._overlay_alpha

    def set_overlay_alpha(self, alpha: float):
        """Set current overlay alpha (for Qt property system)."""
        self._overlay_alpha = alpha
        inference_thread = getattr(self, "inference_thread", None)
        if inference_thread is not None:
            inference_thread.set_overlay_alpha(alpha)

    # Qt property for overlay alpha animation
    overlay_alpha = Property(float, get_overlay_alpha, set_overlay_alpha)

    # ========== Theme Color Animation ==========

    def animate_theme_color(self, target_color: str, duration: int = 150):
        """
        Animate all UI elements to a new theme color.

        Args:
            target_color: Target color as hex string (e.g., "#a60000")
            duration: Animation duration in milliseconds (default: 150ms)
        """
        # Stop existing animation if running
        if self._theme_animation is not None:
            self._theme_animation.stop()

        # Create theme color animation
        self._theme_animation = QPropertyAnimation(self, b"theme_color")
        self._theme_animation.setDuration(duration)
        self._theme_animation.setStartValue(self._current_theme_color)
        self._theme_animation.setEndValue(QColor(target_color))
        self._theme_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._theme_animation.start()

    def get_theme_color(self) -> QColor:
        """Get current theme color (for Qt property system)."""
        return self._current_theme_color

    def set_theme_color(self, color: QColor):
        """Set current theme color and update all UI elements."""
        self._current_theme_color = color
        rgb = f"rgb({color.red()}, {color.green()}, {color.blue()})"

        # Lighten color for borders and accents
        lighter = color.lighter(120)
        rgb_lighter = f"rgb({lighter.red()}, {lighter.green()}, {lighter.blue()})"

        # Darken color for button backgrounds
        darker = color.darker(150)
        rgb_darker = f"rgb({darker.red()}, {darker.green()}, {darker.blue()})"

        # Update all UI elements with themed colors
        stylesheet = f"""
            QWidget#centralWidget {{
                background-color: {rgb};
            }}

            QLabel#image_widget {{
                background-color: #1a1a1a;
                border: 3px solid {rgb_lighter};
                border-radius: 4px;
            }}

            QPushButton {{
                background-color: {rgb_darker};
                border: 2px solid {rgb_lighter};
                border-radius: 4px;
            }}

            QPushButton:hover {{
                background-color: {rgb};
                border: 2px solid {rgb_lighter};
            }}

            QSlider::groove:horizontal {{
                border: 2px solid {rgb_lighter};
                height: 8px;
                background: {rgb_darker};
                margin: 2px 0;
                border-radius: 4px;
            }}

            QSlider::handle:horizontal {{
                background: {rgb_lighter};
                border: 2px solid {rgb_lighter};
                width: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }}

            QProgressBar {{
                border: 2px solid {rgb_lighter};
                border-radius: 2px;
                background-color: {rgb_darker};
            }}

            QProgressBar::chunk {{
                background-color: {rgb_lighter};
            }}
        """

        self.centralWidget().setStyleSheet(stylesheet)
        self.image_widget.setStyleSheet(f"""
            QLabel {{
                background-color: #1a1a1a;
                border: 3px solid {rgb_lighter};
                border-radius: 4px;
            }}
        """)

    # Qt property for theme color animation
    theme_color = Property(QColor, get_theme_color, set_theme_color)

    # ========== Stream Border Color ==========

    def set_stream_border_color(self, color: str, width: int = 5):
        """
        Set border color of livestream (instant, no animation).

        Args:
            color: Border color as hex string (e.g., "#ff0000" for red, "#00ff00" for green)
            width: Border width in pixels (default: 5)
        """
        self.image_widget.setStyleSheet(f"""
            QLabel {{
                background-color: #1a1a1a;
                border: {width}px solid {color};
                border-radius: 4px;
            }}
        """)
