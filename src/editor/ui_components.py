from PyQt5.QtWidgets import (
    QToolBar, QAction, QPushButton, QSlider, QSpinBox,
    QColorDialog, QButtonGroup
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

class UIComponents:
    def __init__(self, main_window):
        self.main_window = main_window
        self.setup_toolbar()
        # Only update cursor if graphics_view exists and current tool is Pencil
        if hasattr(self.main_window, 'graphics_view'):
            self.main_window.graphics_view.update_pencil_cursor()

    def setup_toolbar(self):
        """Create and setup the toolbar with all its components"""
        self.toolbar = QToolBar()
        self.main_window.addToolBar(self.toolbar)

        # Create actions
        self.save_action = QAction("Save", self.main_window)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setToolTip("Save (Ctrl+S)")
        self.smooth_action = QAction("Smooth", self.main_window)
        self.thin_action = QAction("Thin", self.main_window)
        self.smooth_thin_smooth_action = QAction("Smooth-Thin-Smooth", self.main_window)
        self.pencil_action = QAction("Pencil", self.main_window)
        self.eraser_action = QAction("Eraser", self.main_window)
        self.line_action = QAction("Line", self.main_window)
        self.undo_action = QAction("Undo", self.main_window)
        self.undo_action.setToolTip("Undo (Ctrl+Z)")
        self.redo_action = QAction("Redo", self.main_window)
        self.redo_action.setToolTip("Redo (Ctrl+Shift+Z)")

        # Make tool actions checkable
        self.pencil_action.setCheckable(True)
        self.eraser_action.setCheckable(True)
        self.line_action.setCheckable(True)
        self.line_action.setChecked(True)

        # Setup keyboard shortcuts
        self.undo_action.setShortcut("Ctrl+Z")
        self.redo_action.setShortcut("Ctrl+Shift+Z")

        # Add actions to toolbar
        self.toolbar.addAction(self.save_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.pencil_action)
        self.toolbar.addAction(self.line_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.eraser_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.undo_action)
        self.toolbar.addAction(self.redo_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.smooth_action)
        self.toolbar.addAction(self.thin_action)
        self.toolbar.addAction(self.smooth_thin_smooth_action)
        self.toolbar.addSeparator()

        # Add size controls
        self.size_label = QAction("Size:", self.main_window)
        self.toolbar.addAction(self.size_label)

        self.setup_size_controls()
        self.setup_size_presets()
        
        # Add opacity controls
        self.opacity_label = QAction("Opacity:", self.main_window)
        self.toolbar.addAction(self.opacity_label)
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setFixedWidth(100)
        self.toolbar.addWidget(self.opacity_slider)
        
        self.toolbar.addSeparator()
        
        self.setup_color_button()
        self.update_color_button(QColor(255, 165, 0))
        
        # Add zoom controls
        self.toolbar.addSeparator()
        self.zoom_label = QAction("Zoom:", self.main_window)
        self.toolbar.addAction(self.zoom_label)
        
        self.zoom_out_action = QAction("−", self.main_window)
        self.zoom_out_action.setToolTip("Zoom Out (Ctrl+-)")
        self.toolbar.addAction(self.zoom_out_action)
        
        self.zoom_level_label = QAction("100%", self.main_window)
        self.zoom_level_label.setEnabled(False)
        self.toolbar.addAction(self.zoom_level_label)
        
        self.zoom_in_action = QAction("+", self.main_window)
        self.zoom_in_action.setToolTip("Zoom In (Ctrl++)")
        self.toolbar.addAction(self.zoom_in_action)
        
        self.zoom_fit_action = QAction("Fit", self.main_window)
        self.zoom_fit_action.setToolTip("Fit to Window")
        self.toolbar.addAction(self.zoom_fit_action)

        self.zoom_100_action = QAction("100%", self.main_window)
        self.zoom_100_action.setToolTip("Actual Size (Ctrl+0)")
        self.toolbar.addAction(self.zoom_100_action)
        
    def setup_size_controls(self):
        """Setup size spinbox and slider"""
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

        self.toolbar.addSeparator()

    def setup_size_presets(self):
        """Setup preset size buttons"""
        self.size_button_group = QButtonGroup()
        self.size_button_group.setExclusive(True)

        # Define preset sizes
        presets = [4, 8, 16, 32]
        
        for size in presets:
            btn = QPushButton(str(size))
            btn.setCheckable(True)
            btn.setFixedSize(32, 32)
            if size == 8:  # Default size
                btn.setChecked(True)
            self.toolbar.addWidget(btn)
            self.size_button_group.addButton(btn)
            btn.clicked.connect(lambda checked, s=size: self.update_size_controls(s))

        self.toolbar.addSeparator()

    def update_size_controls(self, size):
        """Update all size controls to match the given size"""
        self.size_slider.setValue(size)
        self.size_spinbox.setValue(size)
        
        # Update button states
        for button in self.size_button_group.buttons():
            button.setChecked(int(button.text()) == size)

    def setup_color_button(self):
        """Setup color picker button"""
        self.color_button = QPushButton()
        self.color_button.setFixedSize(32, 32)
        self.toolbar.addWidget(self.color_button)
        # Color will be set when tool is selected

    def update_color_button(self, color):
        """Update color button appearance"""
        self.color_button.setStyleSheet(
            f"background-color: {color.name()}; border: 2px solid #666666;"
        )

    def update_tool_specific_controls(self, tool):
        """Update controls based on tool settings"""
        settings = self.main_window.state.current_tool_settings
        self.update_size_controls(settings["size"])
        self.update_color_button(settings["color"])
        self.color_button.setEnabled(tool != "Eraser")  # Disable color picker for eraser
