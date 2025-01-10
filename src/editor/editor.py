import sys
import signal
import argparse
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from src.editor.drawing_app import DrawingApp

def signal_handler(signum, frame):
    """Handle CTRL+C by closing the Qt application"""
    QApplication.quit()

if __name__ == "__main__":
    # Set up signal handler for CTRL+C
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Drawing Tool with Smoothing")
    parser.add_argument(
        "--input_dir",
        default="./results",
        help="Directory containing input images (default: ./results)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    
    # Allow CTRL+C to interrupt the Qt event loop
    timer = app.startTimer(500)
    def timerEvent(event):
        app.processEvents()
    app.timerEvent = timerEvent

    window = DrawingApp(input_dir=args.input_dir)
    window.show()
    sys.exit(app.exec_())
