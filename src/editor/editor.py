import sys
import argparse
from PyQt5.QtWidgets import (
    QApplication,
)  # Add QSpinBox to imports

from src.editor.drawing_app import DrawingApp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drawing Tool with Smoothing")
    parser.add_argument(
        "--input_dir",
        default="./results",
        help="Directory containing input images (default: ./results)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = DrawingApp(input_dir=args.input_dir)
    window.show()
    sys.exit(app.exec_())
