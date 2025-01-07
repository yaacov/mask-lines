from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor


class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.panning = False
        self.last_point = None
        self.last_pan_pos = None
        self.current_tool = "Pencil"
        self.main_window = parent  # Store reference to main window
        self.temp_line = None  # Add temporary line for preview
        self.setDragMode(QGraphicsView.NoDrag)
        # Enable scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.pen_size = 3
        self.pen_color = QColor(255, 255, 0)  # Initialize with yellow color
        # Add cursor definitions
        self.tool_cursors = {
            "Pencil": Qt.PointingHandCursor,  # Changed from CrossCursor
            "Line": Qt.CrossCursor,
            "Eraser": Qt.CustomCursor,
        }
        # Set default cursor
        self.setCursor(self.tool_cursors["Pencil"])
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus

    def clear_temp_line(self):
        if self.temp_line:
            if self.temp_line.scene():
                self.temp_line.scene().removeItem(self.temp_line)
            self.temp_line = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.scene() and self.main_window.image:
            # Clear any existing temporary line
            self.clear_temp_line()
            self.drawing = True
            self.last_point = self.mapToScene(event.pos())
            if self.current_tool == "Line":
                # Create temporary line item for preview
                self.temp_line = self.scene().addLine(
                    self.last_point.x(),
                    self.last_point.y(),
                    self.last_point.x(),
                    self.last_point.y(),
                    QPen(self.pen_color, self.pen_size, Qt.SolidLine),
                )
        elif event.button() == Qt.MiddleButton:
            self.panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drawing and self.scene() and self.main_window.image:
            current_point = self.mapToScene(event.pos())
            if self.last_point and not current_point.isNull():
                if self.current_tool == "Line":
                    # Update temporary line preview
                    if self.temp_line:
                        self.temp_line.setLine(
                            self.last_point.x(),
                            self.last_point.y(),
                            current_point.x(),
                            current_point.y(),
                        )
                        self.temp_line.setPen(
                            QPen(self.pen_color, self.pen_size, Qt.SolidLine)
                        )
                else:
                    try:
                        painter = QPainter(self.main_window.image)
                        if self.current_tool == "Eraser":
                            # Set composition mode to clear for eraser
                            painter.setCompositionMode(QPainter.CompositionMode_Clear)
                            painter.setPen(
                                QPen(Qt.transparent, self.pen_size, Qt.SolidLine)
                            )
                        else:
                            painter.setCompositionMode(
                                QPainter.CompositionMode_SourceOver
                            )
                            painter.setPen(
                                QPen(self.pen_color, self.pen_size, Qt.SolidLine)
                            )
                        painter.drawLine(
                            self.last_point.toPoint(), current_point.toPoint()
                        )
                        painter.end()
                        self.last_point = current_point
                        self.main_window.update_scene()
                    except Exception as e:
                        print(f"Drawing error: {e}")
                        self.drawing = False
                        self.last_point = None
        elif self.panning and self.last_pan_pos is not None:
            delta = event.pos() - self.last_pan_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self.last_pan_pos = event.pos()
            self.last_point = None
        elif self.panning and self.last_pan_pos is not None:
            delta = event.pos() - self.last_pan_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self.last_pan_pos = event.pos()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing:
                if self.current_tool == "Line" and self.last_point:
                    # Draw the final line on the image
                    try:
                        # Clear temp line first
                        self.clear_temp_line()

                        current_point = self.mapToScene(event.pos())
                        # Draw the final line
                        painter = QPainter(self.main_window.image)
                        pen = QPen(self.pen_color, self.pen_size, Qt.SolidLine)
                        painter.setPen(pen)
                        painter.drawLine(
                            self.last_point.toPoint(), current_point.toPoint()
                        )
                        painter.end()

                        # Update the scene after drawing
                        self.main_window.update_scene()
                    except Exception as e:
                        print(f"Drawing error: {e}")

                # Save state after any drawing operation completes
                self.main_window.save_undo_state()

            # Reset drawing state
            self.drawing = False
            self.last_point = None

        elif event.button() == Qt.MiddleButton:
            self.panning = False
            self.last_pan_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()

    def keyPressEvent(self, event):
        # Handle Ctrl+Z for undo
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            self.main_window.undo()
        super().keyPressEvent(event)
