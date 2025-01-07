import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QGraphicsScene,
    QFileDialog,
    QToolBar,
    QAction,
    QColorDialog,
    QPushButton,
    QSlider,
    QHBoxLayout,
    QListWidget,
    QSpinBox,
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor
from PyQt5.QtCore import Qt
from skimage.filters import gaussian
from skimage.morphology import thin

from .custom_graphics_view import CustomGraphicsView
from .undo_manager import UndoManager
from .image_utils import ImageUtils


class DrawingApp(QMainWindow):
    def __init__(self, input_dir="./results"):
        super().__init__()
        self.setWindowTitle("Drawing Tool with Smoothing")
        self.resize(1280, 800)  # Set initial window size to 1280x800
        self.input_dir = input_dir

        # Initialize central widget with horizontal layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        # Create sidebar for image list
        self.sidebar = QListWidget()
        self.sidebar.setMaximumWidth(200)
        self.layout.addWidget(self.sidebar)

        # Create container for graphics view
        self.drawing_container = QWidget()
        self.drawing_layout = QVBoxLayout(self.drawing_container)
        self.layout.addWidget(self.drawing_container)

        # Initialize graphics view
        self.graphics_view = CustomGraphicsView(self)
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.drawing_layout.addWidget(self.graphics_view)

        # Create toolbar
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)

        # Create actions
        self.save_action = QAction("Save", self)
        self.smooth_action = QAction("Smooth", self)
        self.thin_action = QAction("Thin", self)
        self.pencil_action = QAction("Pencil", self)
        self.eraser_action = QAction("Eraser", self)
        self.line_action = QAction("Line", self)

        # Make tool actions checkable
        self.pencil_action.setCheckable(True)
        self.eraser_action.setCheckable(True)
        self.line_action.setCheckable(True)
        self.pencil_action.setChecked(True)

        # Add actions to toolbar
        self.toolbar.addAction(self.save_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.pencil_action)
        self.toolbar.addAction(self.line_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.eraser_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.smooth_action)
        self.toolbar.addAction(self.thin_action)
        self.toolbar.addSeparator()

        # Add size controls
        self.size_label = QAction("Size:", self)
        self.toolbar.addAction(self.size_label)

        # Add spin box for numeric input
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(4)
        self.size_spinbox.setMaximum(100)
        self.size_spinbox.setValue(8)
        self.size_spinbox.setFixedWidth(60)
        self.toolbar.addWidget(self.size_spinbox)

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(4)
        self.size_slider.setMaximum(100)
        self.size_slider.setValue(8)
        self.size_slider.setFixedWidth(100)
        self.toolbar.addWidget(self.size_slider)

        # Connect size controls to sync with each other
        self.size_slider.valueChanged.connect(self.size_spinbox.setValue)
        self.size_spinbox.valueChanged.connect(self.size_slider.setValue)
        self.size_slider.valueChanged.connect(self.change_pen_size)
        self.size_spinbox.valueChanged.connect(self.change_pen_size)

        # Now we can safely set the pen size
        self.graphics_view.pen_size = 8

        self.toolbar.addSeparator()

        # Add color picker button
        self.color_button = QPushButton()
        self.color_button.setFixedSize(32, 32)
        self.color_button.clicked.connect(self.choose_color)
        self.update_color_button(QColor(255, 255, 0))  # Initialize with yellow color
        self.toolbar.addWidget(self.color_button)

        # Tool selection
        self.current_tool = "Pencil"

        # Connect actions
        self.save_action.triggered.connect(self.save_image)
        self.smooth_action.triggered.connect(self.smooth_image)
        self.thin_action.triggered.connect(self.thin_image)
        self.pencil_action.triggered.connect(lambda: self.select_tool("Pencil"))
        self.eraser_action.triggered.connect(lambda: self.select_tool("Eraser"))
        self.line_action.triggered.connect(lambda: self.select_tool("Line"))

        # Image and drawing setup
        self.image = None
        self.background_image = None
        self.pixmap_item = None
        self.background_item = None
        self.current_filename = "mask-out.png"  # Add default filename

        # Load images from input directory
        self.load_images_list()

        # Connect sidebar clicks
        self.sidebar.itemClicked.connect(self.load_background_from_list)

        # Create input and target directories if they don't exist
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

        inputs_dir = os.path.join(self.input_dir, "inputs")
        targets_dir = os.path.join(self.input_dir, "target")  # Add target directory
        if not os.path.exists(inputs_dir):
            os.makedirs(inputs_dir)
        if not os.path.exists(targets_dir):  # Create target directory
            os.makedirs(targets_dir)

        # Replace existing undo setup with new UndoManager
        self.undo_manager = UndoManager()

        # Create redo action in addition to undo
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut("Ctrl+Shift+Z")
        self.redo_action.triggered.connect(self.redo)

        self.toolbar.insertAction(self.smooth_action, self.undo_action)
        self.toolbar.insertAction(self.smooth_action, self.redo_action)
        self.toolbar.insertSeparator(self.smooth_action)

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

    def load_background(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Background Image", "", "Images (*.png *.jpg *.bmp)"
        )
        if file_path:
            self.background_image = QImage(file_path).convertToFormat(
                QImage.Format_RGB32
            )
            # If no drawing layer exists, create a transparent one
            if self.image is None:
                self.image = QImage(self.background_image.size(), QImage.Format_ARGB32)
                self.undo_stack.pop(0)
        if self.scene:
            self.scene.clear()
            # Add background layer if exists
            if self.background_image:
                bg_pixmap = QPixmap.fromImage(self.background_image)
            self.undo_stack.append(self.image.copy())
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)
        if self.scene:
            self.scene.clear()
            # Add background layer if exists
            if self.background_image:
                bg_pixmap = QPixmap.fromImage(self.background_image)
                self.background_item = self.scene.addPixmap(bg_pixmap)
            # Add drawing layer if exists
            if self.image:
                pixmap = QPixmap.fromImage(self.image)
                self.pixmap_item = self.scene.addPixmap(pixmap)

    def load_image(self):
        """Load mask image from file dialog"""
        file_path = self.get_file_path("Open Mask Image")
        if file_path:
            self.load_image_from_path(file_path)
            self.update_scene()

    def save_image(self):
        if self.image:
            # Use current_filename if set, otherwise use default
            save_filename = (
                self.current_filename
                if hasattr(self, "current_filename")
                else "mask-out.png"
            )
            file_path = os.path.join(self.input_dir, "target", save_filename)

            # Convert to yellow and black based on alpha channel
            image_array = ImageUtils.qimage_to_numpy(self.image)
            # Create a new image where alpha > 0 becomes yellow (255,255,255) and alpha = 0 becomes black (0,0,0)
            bw_array = np.zeros(
                (image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8
            )
            bw_array[image_array[:, :, 3] > 0] = [255, 255, 255]

            # Convert back to QImage and save
            bw_image = QImage(
                bw_array.data,
                bw_array.shape[1],
                bw_array.shape[0],
                bw_array.shape[1] * 3,
                QImage.Format_RGB888,
            )
            bw_image.save(file_path)

    def select_tool(self, tool):
        # Clear any temporary line when changing tools
        self.graphics_view.clear_temp_line()
        self.current_tool = tool
        self.graphics_view.current_tool = tool

        # Update action states
        actions = {
            "Pencil": self.pencil_action,
            "Eraser": self.eraser_action,
            "Line": self.line_action,
        }

        for t, action in actions.items():
            action.setChecked(t == tool)

        # Set the appropriate cursor for the selected tool
        if tool == "Eraser":
            # Create a custom cursor for eraser that looks like a circle
            size = self.graphics_view.pen_size
            pixmap = QPixmap(size + 2, size + 2)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.black, 1))
            painter.drawEllipse(1, 1, size - 1, size - 1)
            painter.end()
            self.graphics_view.setCursor(QCursor(pixmap))
        else:
            # Use the cursor defined in tool_cursors for other tools
            self.graphics_view.setCursor(self.graphics_view.tool_cursors[tool])

    def smooth_image(self):
        if self.image:
            self.save_undo_state()
            # Convert to numpy array
            image_array = ImageUtils.qimage_to_numpy(self.image)

            # Create binary mask based on alpha channel (1 where alpha > 0)
            mask = (image_array[:, :, 3] > 0).astype(np.float32)

            # Apply Gaussian smoothing to the mask
            smoothed_mask = gaussian(mask, sigma=1)

            # Create new RGBA array
            height, width = mask.shape
            result = np.zeros((height, width, 4), dtype=np.uint8)

            # Where smoothed mask > 0, set the color and alpha
            mask_threshold = smoothed_mask > 0.1
            result[mask_threshold] = [
                self.graphics_view.pen_color.red(),
                self.graphics_view.pen_color.green(),
                self.graphics_view.pen_color.blue(),
                255,
            ]

            # Convert back to QImage and update
            self.image = QImage(
                result.data, width, height, width * 4, QImage.Format_RGBA8888
            ).copy()
            self.update_scene()

    def thin_image(self):
        if self.image:
            self.save_undo_state()
            # Convert to numpy array
            image_array = ImageUtils.qimage_to_numpy(self.image)

            # Create binary mask based on alpha channel
            mask = (image_array[:, :, 3] > 0).astype(np.uint8)

            # Apply thinning
            thinned_mask = thin(mask)

            # Create new RGBA array
            height, width = mask.shape
            result = np.zeros((height, width, 4), dtype=np.uint8)

            # Where thinned mask is True, set the color and alpha
            result[thinned_mask] = [
                self.graphics_view.pen_color.red(),
                self.graphics_view.pen_color.green(),
                self.graphics_view.pen_color.blue(),
                255,
            ]

            # Convert back to QImage and update
            self.image = QImage(
                result.data, width, height, width * 4, QImage.Format_RGBA8888
            ).copy()
            self.update_scene()

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.update_color_button(color)
            self.graphics_view.pen_color = color

    def update_color_button(self, color):
        self.color_button.setStyleSheet(
            f"background-color: {color.name()}; border: 2px solid #666666;"
        )

    def change_pen_size(self, size):
        self.graphics_view.pen_size = size

    def load_images_list(self):
        """Load image files from input directory into sidebar"""
        self.sidebar.clear()

        # Ensure both directories exist
        inputs_dir = os.path.join(self.input_dir, "inputs")
        targets_dir = os.path.join(self.input_dir, "target")

        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)
        if not os.path.exists(inputs_dir):
            os.makedirs(inputs_dir)
        if not os.path.exists(targets_dir):
            os.makedirs(targets_dir)

        # Load only files that exist in inputs directory
        for filename in os.listdir(inputs_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                input_path = os.path.join(inputs_dir, filename)
                if os.path.exists(input_path):
                    self.sidebar.addItem(filename)

    def load_background_from_list(self, item):
        """Load background image and corresponding drawing when clicked in sidebar"""
        if not item:
            return

        # Clear undo/redo history before loading new image
        self.undo_manager = UndoManager()

        self.current_filename = item.text()  # Store the filename
        bg_filepath = os.path.join(self.input_dir, "inputs", self.current_filename)
        drawing_filepath = os.path.join(self.input_dir, "target", self.current_filename)

        # Load background using helper method
        self.load_background_from_path(bg_filepath)

        # Load target drawing if exists, otherwise create blank transparent layer
        if os.path.exists(drawing_filepath):
            self.load_image_from_path(drawing_filepath)
        else:
            self.image = QImage(self.background_image.size(), QImage.Format_ARGB32)
            self.image.fill(Qt.transparent)

        # Save initial state to undo manager
        self.save_undo_state()

        self.update_scene()

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

    def update_scene(self):
        """Update the graphics scene with current image and background layers"""
        if self.scene:
            self.scene.clear()

            # Add background layer if exists
            if self.background_image:
                bg_pixmap = QPixmap.fromImage(self.background_image)
                self.background_item = self.scene.addPixmap(bg_pixmap)

            # Add drawing layer if exists
            if self.image:
                pixmap = QPixmap.fromImage(self.image)
                self.pixmap_item = self.scene.addPixmap(pixmap)

            # Fit the view to the scene content
            self.graphics_view.setSceneRect(self.scene.itemsBoundingRect())
