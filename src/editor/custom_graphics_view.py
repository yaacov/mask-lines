from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtGui import QPainter, QPen, QPixmap, QCursor, QTransform
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor


class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.panning = False
        self.last_point = None
        self.last_pan_pos = None
        self.current_tool = "Line"
        self.main_window = parent
        self.temp_line = None
        self.setDragMode(QGraphicsView.NoDrag)
        # Enable scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.pen_size = 8 
        self.pen_color = QColor(255, 165, 0)  # Orange color (RGB: 255, 165, 0)
        # Add cursor definitions
        self.tool_cursors = {
            "Pencil": Qt.CustomCursor, 
            "Line": Qt.CrossCursor,
            "Eraser": Qt.CustomCursor,
        }
        self.update_tool_cursor()  
        self.setFocusPolicy(Qt.StrongFocus) 
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        # Enable mouse tracking for wheel events
        self.setMouseTracking(True)

    def clear_temp_line(self):
        if self.temp_line:
            if self.temp_line.scene():
                self.temp_line.scene().removeItem(self.temp_line)
            self.temp_line = None

    def update_pencil_cursor(self):
        """Update the pencil cursor size"""
        if self.current_tool == "Pencil":
            size = self.pen_size + 2
            pixmap = QPixmap(size + 2, size + 2)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            
            # Draw circle
            painter.setPen(QPen(Qt.black, 1))
            painter.drawEllipse(1, 1, size - 1, size - 1)
            
            # Draw cross
            center = size // 2 + 1
            cross_size = min(size // 3, 4)  # Limit cross size
            painter.drawLine(center - cross_size, center, center + cross_size, center)
            painter.drawLine(center, center - cross_size, center, center + cross_size)
            
            painter.end()
            self.setCursor(QCursor(pixmap))

    def update_eraser_cursor(self):
        """Update the eraser cursor size"""
        if self.current_tool == "Eraser":
            size = self.pen_size
            pixmap = QPixmap(size + 2, size + 2)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.black, 1))
            painter.drawEllipse(1, 1, size - 1, size - 1)
            painter.end()
            self.setCursor(QCursor(pixmap))

    def update_tool_cursor(self):
        """Update cursor based on current tool"""
        if self.current_tool == "Pencil":
            self.update_pencil_cursor()
        elif self.current_tool == "Eraser":
            self.update_eraser_cursor()
        elif self.current_tool == "Line":
            self.setCursor(self.tool_cursors["Line"])

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
                        pen = QPen(self.pen_color, self.pen_size, Qt.SolidLine)
                        pen.setCapStyle(Qt.RoundCap)
                        pen.setJoinStyle(Qt.RoundJoin)
                        self.temp_line.setLine(
                            self.last_point.x(),
                            self.last_point.y(),
                            current_point.x(),
                            current_point.y(),
                        )
                        self.temp_line.setPen(pen)
                else:
                    try:
                        painter = QPainter(self.main_window.image)
                        if self.current_tool == "Eraser":
                            # Set composition mode to clear for eraser
                            painter.setCompositionMode(QPainter.CompositionMode_Clear)
                            pen = QPen(Qt.transparent, self.pen_size, Qt.SolidLine)
                        else:
                            painter.setCompositionMode(
                                QPainter.CompositionMode_SourceOver
                            )
                            pen = QPen(self.pen_color, self.pen_size, Qt.SolidLine)
                        
                        # Set round cap and join for smoother lines
                        pen.setCapStyle(Qt.RoundCap)
                        pen.setJoinStyle(Qt.RoundJoin)
                        painter.setPen(pen)
                        
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
                        pen.setCapStyle(Qt.RoundCap)
                        pen.setJoinStyle(Qt.RoundJoin)
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
        if event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_Z:
                self.main_window.undo()
            elif event.key() == Qt.Key_S:
                self.main_window.save_image()
            elif event.key() == Qt.Key_0:  # Add Ctrl+0 shortcut
                self.zoom_100()
        super().keyPressEvent(event)

    def zoom_in(self):
        """Zoom in by 20%"""
        self.scale_view(1.2)

    def zoom_out(self):
        """Zoom out by 20%"""
        self.scale_view(1/1.2)

    def zoom_fit(self):
        """Fit view to scene contents"""
        if self.scene():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
            # Update zoom factor based on current transform
            self.zoom_factor = self.transform().m11()
            self.update_zoom_label()

    def zoom_100(self):
        """Reset zoom to 100%"""
        self.zoom_factor = 1.0
        transform = QTransform()
        transform.scale(1.0, 1.0)
        self.setTransform(transform)
        self.update_zoom_label()

    def scale_view(self, factor):
        """Scale the view by the given factor"""
        new_zoom = self.zoom_factor * factor
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.zoom_factor = new_zoom
            transform = QTransform()
            transform.scale(self.zoom_factor, self.zoom_factor)
            self.setTransform(transform)
            self.update_zoom_label()

    def update_zoom_label(self):
        """Update the zoom level display in the toolbar"""
        if hasattr(self.main_window, 'ui'):
            zoom_percent = int(self.zoom_factor * 100)
            self.main_window.ui.zoom_level_label.setText(f"{zoom_percent}%")

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if event.modifiers() == Qt.ControlModifier:
            # Get the position before scaling to use as anchor point
            anchor_pos = self.mapToScene(event.pos())

            # Calculate zoom factor based on wheel delta
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else 1/1.1
            
            # Check if the new zoom would be within bounds
            new_zoom = self.zoom_factor * factor
            if self.min_zoom <= new_zoom <= self.max_zoom:
                self.zoom_factor = new_zoom
                # Apply the transform
                transform = QTransform()
                transform.scale(self.zoom_factor, self.zoom_factor)
                self.setTransform(transform)
                
                # Adjust the view to maintain the point under cursor
                new_pos = self.mapFromScene(anchor_pos)
                delta = new_pos - event.pos()
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
                
                # Update zoom display
                self.update_zoom_label()
            
            event.accept()
        else:
            # Handle normal scrolling
            super().wheelEvent(event)
