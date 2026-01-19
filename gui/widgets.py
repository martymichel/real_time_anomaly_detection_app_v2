"""Custom widgets for anomaly detection GUI."""
from PySide6.QtWidgets import QSlider, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor


class MarkedSlider(QSlider):
    """
    QSlider with a vertical marker line at a reference position.

    Used to show the original trained threshold value while allowing
    the user to adjust the current threshold via the slider.
    """

    def __init__(self, orientation: Qt.Orientation = Qt.Orientation.Horizontal, parent: QWidget = None):
        super().__init__(orientation, parent)
        self._marker_value: float | None = None
        self._marker_color = QColor("#ffaa00")  # Orange/amber color for visibility
        self._marker_width = 2

    def set_marker_value(self, value: float | None):
        """
        Set the marker position (in slider value units, e.g., 0-1000).

        Args:
            value: The slider value where to show the marker, or None to hide.
        """
        self._marker_value = value
        self.update()  # Trigger repaint

    def get_marker_value(self) -> float | None:
        """Get the current marker value."""
        return self._marker_value

    def set_marker_color(self, color: QColor | str):
        """Set the marker line color."""
        if isinstance(color, str):
            color = QColor(color)
        self._marker_color = color
        self.update()

    def paintEvent(self, event):
        """Override paint event to draw marker line."""
        # First, let the base class draw the slider
        super().paintEvent(event)

        # Draw marker if set
        if self._marker_value is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Calculate marker position
            slider_min = self.minimum()
            slider_max = self.maximum()
            slider_range = slider_max - slider_min

            if slider_range <= 0:
                return

            # Get the groove geometry (where the slider track is)
            # Account for handle width to get accurate position
            handle_width = 18  # From stylesheet
            available_width = self.width() - handle_width
            offset = handle_width // 2

            # Calculate x position for the marker
            ratio = (self._marker_value - slider_min) / slider_range
            x_pos = int(offset + ratio * available_width)

            # Draw the marker line
            pen = QPen(self._marker_color)
            pen.setWidth(self._marker_width)
            painter.setPen(pen)

            # Draw vertical line through the groove area
            groove_top = 6  # Approximate top of groove
            groove_bottom = self.height() - 6  # Approximate bottom of groove

            painter.drawLine(x_pos, groove_top, x_pos, groove_bottom)

            painter.end()
