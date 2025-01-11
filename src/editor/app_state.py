from PyQt5.QtGui import QColor, QImage
from PyQt5.QtCore import QObject, pyqtSignal

class AppState(QObject):
    # Signals for state changes
    tool_changed = pyqtSignal(str)
    pen_size_changed = pyqtSignal(int)
    pen_color_changed = pyqtSignal(QColor)
    opacity_changed = pyqtSignal(float)
    zoom_changed = pyqtSignal(float)
    image_changed = pyqtSignal()
    unsaved_changes = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._current_tool = "Line"
        self._tool_settings = {
            "Pencil": {
                "size": 8,
                "color": QColor(255, 165, 0)  # Orange
            },
            "Line": {
                "size": 8,
                "color": QColor(255, 165, 0)  # Orange
            },
            "Eraser": {
                "size": 32,
                "color": QColor(0, 0, 0, 0)  # Transparent
            }
        }
        self._opacity = 0.5
        self._zoom_factor = 1.0
        self._current_image = None
        self._background_image = None
        self._has_unsaved_changes = False
        self._current_filename = "mask-out.png"

    # Tool property
    @property
    def current_tool(self):
        return self._current_tool

    @current_tool.setter
    def current_tool(self, tool):
        if self._current_tool != tool:
            self._current_tool = tool
            self.tool_changed.emit(tool)

    @property
    def current_tool_settings(self):
        return self._tool_settings[self._current_tool]

    def update_tool_setting(self, tool, setting, value):
        if tool in self._tool_settings and setting in self._tool_settings[tool]:
            self._tool_settings[tool][setting] = value
            if tool == self._current_tool:
                if setting == "size":
                    self.pen_size_changed.emit(value)
                elif setting == "color":
                    self.pen_color_changed.emit(value)

    def save_tool_state(self, tool, settings):
        """Save current settings for the given tool"""
        if tool in self._tool_settings:
            self._tool_settings[tool] = settings.copy()

    def restore_tool_state(self, tool):
        """Restore saved settings for the given tool"""
        if tool in self._tool_settings:
            return self._tool_settings[tool].copy()
        return None

    # Pen size property
    @property
    def pen_size(self):
        return self._pen_size

    @pen_size.setter
    def pen_size(self, size):
        if self._pen_size != size:
            self._pen_size = size
            self.pen_size_changed.emit(size)

    # Pen color property
    @property
    def pen_color(self):
        return self._pen_color

    @pen_color.setter
    def pen_color(self, color):
        if self._pen_color != color:
            self._pen_color = color
            self.pen_color_changed.emit(color)

    # Opacity property
    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        if self._opacity != value:
            self._opacity = value
            self.opacity_changed.emit(value)

    # Zoom factor property
    @property
    def zoom_factor(self):
        return self._zoom_factor

    @zoom_factor.setter
    def zoom_factor(self, factor):
        if self._zoom_factor != factor:
            self._zoom_factor = factor
            self.zoom_changed.emit(factor)

    # Current image property
    @property
    def current_image(self):
        return self._current_image

    @current_image.setter
    def current_image(self, image):
        self._current_image = image
        self.image_changed.emit()

    # Background image property
    @property
    def background_image(self):
        return self._background_image

    @background_image.setter
    def background_image(self, image):
        self._background_image = image
        self.image_changed.emit()

    # Current filename property
    @property
    def current_filename(self):
        return self._current_filename

    @current_filename.setter
    def current_filename(self, filename):
        self._current_filename = filename

    # Unsaved changes property
    @property
    def has_unsaved_changes(self):
        return self._has_unsaved_changes

    @has_unsaved_changes.setter
    def has_unsaved_changes(self, value):
        if self._has_unsaved_changes != value:
            self._has_unsaved_changes = value
            self.unsaved_changes.emit(value)

    def create_blank_image(self, size):
        """Create a new blank transparent image"""
        image = QImage(size, QImage.Format_ARGB32)
        image.fill(QColor(0, 0, 0, 0))
        return image
