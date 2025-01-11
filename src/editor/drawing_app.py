import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QGraphicsScene,
    QHBoxLayout, QListWidget, QFileDialog, QColorDialog
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt

from .custom_graphics_view import CustomGraphicsView
from .undo_manager import UndoManager
from .image_utils import ImageUtils
from .ui_components import UIComponents
from .image_processor import ImageProcessor
from .app_state import AppState


class DrawingApp(QMainWindow):
    def __init__(self, input_dir="./results"):
        super().__init__()
        # Initialize state manager
        self.state = AppState()
        # Connect state signals
        self.connect_state_signals()
        self.init_window()
        self.init_attributes(input_dir)
        self.init_ui_components()
        self.init_directories()
        self.load_images_list()

    # ---- Initialization Methods ----
    
    def init_window(self):
        """Initialize window properties"""
        self.setWindowTitle("Drawing Tool with Smoothing")
        self.resize(1280, 800)
        
        # Initialize central widget with horizontal layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

    def init_attributes(self, input_dir):
        """Initialize class attributes"""
        self.input_dir = input_dir
        self.current_tool = "Line"
        self.image = None
        self.background_image = None
        self.pixmap_item = None
        self.background_item = None
        self.current_filename = "mask-out.png"
        self.has_unsaved_changes = False
        self.undo_manager = UndoManager()

    def init_ui_components(self):
        """Initialize UI components and layouts"""
        # Create sidebar
        self.setup_sidebar()
        
        # Create drawing area
        self.setup_drawing_area()
        
        # Initialize UI components and connect actions
        self.ui = UIComponents(self)
        self.connect_actions()

    def setup_sidebar(self):
        """Setup sidebar for image list"""
        self.sidebar = QListWidget()
        self.sidebar.setMaximumWidth(200)
        self.sidebar.itemClicked.connect(self.load_background_from_list)
        self.layout.addWidget(self.sidebar)

    def setup_drawing_area(self):
        """Setup drawing container and graphics view"""
        self.drawing_container = QWidget()
        self.drawing_layout = QVBoxLayout(self.drawing_container)
        self.layout.addWidget(self.drawing_container)

        self.graphics_view = CustomGraphicsView(self)
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.drawing_layout.addWidget(self.graphics_view)

    def init_directories(self):
        """Initialize required directories"""
        directories = [
            self.input_dir,
            os.path.join(self.input_dir, "inputs"),
            os.path.join(self.input_dir, "targets")
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    # ---- Action Connections ----

    def connect_actions(self):
        """Connect all UI actions to their handlers"""
        # File actions
        self.ui.save_action.triggered.connect(self.save_image)
        
        # Tool actions
        self.ui.pencil_action.triggered.connect(lambda: self.select_tool("Pencil"))
        self.ui.eraser_action.triggered.connect(lambda: self.select_tool("Eraser"))
        self.ui.line_action.triggered.connect(lambda: self.select_tool("Line"))
        
        # Edit actions
        self.ui.undo_action.triggered.connect(self.undo)
        self.ui.redo_action.triggered.connect(self.redo)
        self.ui.smooth_action.triggered.connect(self.smooth_image)
        self.ui.thin_action.triggered.connect(self.thin_image)
        self.ui.smooth_thin_smooth_action.triggered.connect(self.smooth_thin_smooth_image)  # Add this line
        
        # Style actions
        self.ui.color_button.clicked.connect(self.choose_color)
        
        # Size control connections
        self.connect_size_controls()

        # Opacity control connection
        self.ui.opacity_slider.valueChanged.connect(self.change_opacity)
        
        # Connect zoom controls
        self.ui.zoom_in_action.triggered.connect(self.graphics_view.zoom_in)
        self.ui.zoom_out_action.triggered.connect(self.graphics_view.zoom_out)
        self.ui.zoom_fit_action.triggered.connect(self.graphics_view.zoom_fit)
        self.ui.zoom_100_action.triggered.connect(self.graphics_view.zoom_100)
        
        # Add keyboard shortcuts for zoom
        self.ui.zoom_in_action.setShortcut("Ctrl++")
        self.ui.zoom_out_action.setShortcut("Ctrl+-")
        self.ui.zoom_100_action.setShortcut("Ctrl+0")

    def connect_size_controls(self):
        """Connect size-related controls"""
        self.ui.size_slider.valueChanged.connect(self.ui.size_spinbox.setValue)
        self.ui.size_spinbox.valueChanged.connect(self.ui.size_slider.setValue)
        self.ui.size_slider.valueChanged.connect(self.change_pen_size)
        self.ui.size_spinbox.valueChanged.connect(self.change_pen_size)
        self.ui.size_slider.valueChanged.connect(self.ui.update_size_controls)
        self.ui.size_spinbox.valueChanged.connect(self.ui.update_size_controls)

    def connect_state_signals(self):
        """Connect state change signals to UI updates"""
        self.state.image_changed.connect(self.update_scene)
        self.state.unsaved_changes.connect(self.update_window_title)

    # ---- File Operations ----

    def load_images_list(self):
        """Load image files from input directory into sidebar"""
        self.sidebar.clear()
        inputs_dir = os.path.join(self.input_dir, "inputs")
        
        if os.path.exists(inputs_dir):
            for filename in os.listdir(inputs_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    input_path = os.path.join(inputs_dir, filename)
                    if os.path.exists(input_path):
                        self.sidebar.addItem(filename)

    def get_file_path(self, title="Open Image", directory=""):
        """Helper method to get file path from dialog"""
        return QFileDialog.getOpenFileName(
            self, title, directory, "Images (*.png *.jpg *.bmp)"
        )[0]

    def load_image_from_path(self, file_path):
        """Helper method to load and process image from path"""
        if not file_path or not os.path.exists(file_path):
            return

        # Load image and convert to numpy array
        temp_image = QImage(file_path)
        array = ImageUtils.qimage_to_numpy(temp_image)

        # Convert to grayscale
        grayscale = np.mean(array[:, :, :3], axis=2)

        # Threshold to black and white (binary)
        binary = (grayscale > 128).astype(np.uint8)

        # Create RGBA array
        height, width = binary.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)

        # Set yellow color (255, 255, 0) where binary is 1
        rgba[binary == 1] = [255, 255, 0, 255]

        # Convert back to QImage
        self.image = QImage(
            rgba.data, width, height, width * 4, QImage.Format_RGBA8888
        ).copy()

        if self.background_image and (
            self.image.size() != self.background_image.size()
        ):
            # Resize drawing layer to match background
            self.image = self.image.scaled(self.background_image.size())

    def load_background_from_path(self, file_path):
        """Helper method to load background from path"""
        if not file_path or not os.path.exists(file_path):
            return

        self.background_image = QImage(file_path).convertToFormat(QImage.Format_RGB32)
        # If no drawing layer exists, create a transparent one
        if self.image is None:
            self.image = QImage(self.background_image.size(), QImage.Format_ARGB32)
            self.image.fill(Qt.transparent)
        self.update_scene()

    def save_image(self):
        if self.image:
            save_filename = (
                self.current_filename
                if hasattr(self, "current_filename")
                else "mask-out.png"
            )
            file_path = os.path.join(self.input_dir, "targets", save_filename)
            
            # Convert to binary image and save
            binary_image = ImageProcessor.create_binary_save_image(self.image)
            if binary_image:
                binary_image.save(file_path)
                self.has_unsaved_changes = False
                self.update_window_title()

                # Update the image view with yellow pixels
                array = ImageUtils.qimage_to_numpy(self.image)
                height, width = array.shape[:2]
                rgba = np.zeros((height, width, 4), dtype=np.uint8)
                
                # Set yellow color (255, 255, 0) where alpha channel is not 0
                mask = array[:, :, 3] > 0
                rgba[mask] = [255, 255, 0, 255]
                
                # Convert back to QImage and update the view
                self.image = QImage(
                    rgba.data, width, height, width * 4, QImage.Format_RGBA8888
                ).copy()
                self.update_scene()

    def load_background_from_list(self, item):
        """Load background image and corresponding drawing when clicked in sidebar"""
        if not item:
            return

        # Clear undo/redo history before loading new image
        self.undo_manager = UndoManager()

        self.current_filename = item.text()  # Store the filename
        self.has_unsaved_changes = False
        self.update_window_title()
        bg_filepath = os.path.join(self.input_dir, "inputs", self.current_filename)
        drawing_filepath = os.path.join(self.input_dir, "targets", self.current_filename)

        # Load background using helper method
        self.load_background_from_path(bg_filepath)

        # Load targets drawing if exists, otherwise create blank transparent layer
        if os.path.exists(drawing_filepath):
            self.load_image_from_path(drawing_filepath)
        else:
            self.image = QImage(self.background_image.size(), QImage.Format_ARGB32)
            self.image.fill(Qt.transparent)

        # Add initial state to undo manager without marking as unsaved
        self.undo_manager.push(self.image.copy())
        
        self.update_scene()

    # ---- Tool Operations ----

    def select_tool(self, tool):
        """Select and configure the current drawing tool"""
        # Save current tool settings before switching
        if self.current_tool:
            self.state.save_tool_state(self.current_tool, {
                "size": self.graphics_view.pen_size,
                "color": self.graphics_view.pen_color
            })
        
        self.graphics_view.clear_temp_line()
        self.current_tool = tool
        self.graphics_view.current_tool = tool

        # Restore saved settings for the selected tool
        tool_settings = self.state.restore_tool_state(tool)
        if tool_settings:
            self.graphics_view.pen_size = tool_settings["size"]
            self.graphics_view.pen_color = tool_settings["color"]
            
            # Update UI with tool settings
            self.ui.update_size_controls(tool_settings["size"])
            self.ui.update_color_button(tool_settings["color"])

        # Update action states
        actions = {
            "Pencil": self.ui.pencil_action,
            "Eraser": self.ui.eraser_action,
            "Line": self.ui.line_action,
        }
        for t, action in actions.items():
            action.setChecked(t == tool)

        # Update cursor
        self.update_tool_cursor(tool)

    def update_tool_cursor(self, tool):
        """Update cursor based on selected tool"""
        if tool == "Eraser":
            self.graphics_view.update_eraser_cursor()
        elif tool == "Pencil":
            self.graphics_view.update_pencil_cursor()
        else:
            self.graphics_view.setCursor(self.graphics_view.tool_cursors[tool])

    # ---- Image Processing Operations ----

    def smooth_image(self):
        """Apply smoothing to the current image"""
        if self.image:
            self.save_undo_state()
            self.image = ImageProcessor.smooth_image(self.image, self.graphics_view.pen_color)
            self.update_scene()

    def thin_image(self):
        """Apply thinning to the current image"""
        if self.image:
            self.save_undo_state()
            self.image = ImageProcessor.thin_image(self.image, self.graphics_view.pen_color)
            self.update_scene()

    def smooth_thin_smooth_image(self):
        """Apply smooth-thin-smooth filter to the current image"""
        if self.image:
            self.save_undo_state()
            self.image = ImageProcessor.smooth_thin_smooth_image(self.image, self.graphics_view.pen_color)
            self.update_scene()

    # ---- Style Operations ----

    def choose_color(self):
        """Open color picker dialog"""
        if self.current_tool != "Eraser":  # Prevent changing eraser color
            color = QColorDialog.getColor()
            if color.isValid():
                self.state.update_tool_setting(self.current_tool, "color", color)
                self.ui.update_color_button(color)
                self.graphics_view.pen_color = color

    def update_color_button(self, color):
        """Deprecated: Use ui.update_color_button instead"""
        self.ui.update_color_button(color)

    def change_pen_size(self, size):
        """Update pen size and cursor"""
        # Store size in current tool's settings
        self.state.update_tool_setting(self.current_tool, "size", size)
        self.graphics_view.pen_size = size
        if self.current_tool in ["Eraser", "Pencil"]:
            getattr(self.graphics_view, f"update_{self.current_tool.lower()}_cursor")()

    def change_opacity(self, value):
        """Update the opacity of the drawing layer"""
        if self.pixmap_item:
            opacity = value / 100.0
            self.pixmap_item.setOpacity(opacity)

    # ---- Scene Management ----

    def update_scene(self):
        """Update the graphics scene with current layers"""
        if not self.scene:
            return

        self.scene.clear()
        
        # Add background layer
        if self.background_image:
            bg_pixmap = QPixmap.fromImage(self.background_image)
            self.background_item = self.scene.addPixmap(bg_pixmap)

        # Add drawing layer with opacity
        if self.image:
            pixmap = QPixmap.fromImage(self.image)
            self.pixmap_item = self.scene.addPixmap(pixmap)
            opacity = self.ui.opacity_slider.value() / 100.0
            self.pixmap_item.setOpacity(opacity)

        # Update view
        self.graphics_view.setSceneRect(self.scene.itemsBoundingRect())
        
        # Ensure zoom label is updated after scene changes
        self.graphics_view.update_zoom_label()

    # ---- State Management ----

    def undo(self):
        """Handle undo request"""
        if not self.undo_manager.can_undo():
            return

        prev_state = self.undo_manager.undo()
        if prev_state:
            self.image = prev_state
            self.update_scene()

    def redo(self):
        """Handle redo request"""
        state = self.undo_manager.redo()
        if state:
            self.image = state.copy()
            self.update_scene()

    def save_undo_state(self):
        """Save current state for undo"""
        if self.image:
            self.undo_manager.push(self.image.copy())
            self.has_unsaved_changes = True
            self.update_window_title()

    def update_window_title(self):
        """Update window title with filename and save status"""
        title = f"Drawing Tool - {self.state.current_filename}"
        if self.state.has_unsaved_changes:
            title += " *"
        self.setWindowTitle(title)


    def update_window_title(self):
        """Update window title with filename and save status"""
        title = f"Drawing Tool - {self.state.current_filename}"
        if self.state.has_unsaved_changes:
            title += " *"
        self.setWindowTitle(title)
