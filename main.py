import sys
import cv2
import csv
import pytesseract
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QComboBox,
    QLineEdit, QCheckBox
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PIL import Image
import numpy as np

# Uncomment and adjust path if needed (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCRScanner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Claims Scanner v0.0.1")
        self.setGeometry(100, 100, 800, 600)

        self.video_capture = None
        self.current_frame = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.extracted_data = []  # Stores all scanned entries

        self.init_ui()
        self.detect_cameras()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.camera_selector = QComboBox()
        self.camera_selector.currentIndexChanged.connect(self.change_camera)
        layout.addWidget(self.camera_selector)

        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Flip controls for camera orientation
        transform_layout = QHBoxLayout()
        self.flip_horizontal_checkbox = QCheckBox("Flip horizontally")
        self.flip_vertical_checkbox = QCheckBox("Flip vertically")
        transform_layout.addWidget(self.flip_horizontal_checkbox)
        transform_layout.addWidget(self.flip_vertical_checkbox)
        layout.addLayout(transform_layout)

        self.capture_button = QPushButton("Capture & OCR")
        self.capture_button.clicked.connect(self.capture_and_ocr)
        layout.addWidget(self.capture_button)

        # File path row: input + browse
        path_layout = QHBoxLayout()
        self.path_label = QLabel("Save to:")
        path_layout.addWidget(self.path_label)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select a file to save or append CSV data...")
        path_layout.addWidget(self.path_edit)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_save_path)
        path_layout.addWidget(self.browse_button)
        layout.addLayout(path_layout)

        self.append_checkbox = QCheckBox("Append to existing file")
        self.append_checkbox.setChecked(True)
        layout.addWidget(self.append_checkbox)

        central_widget.setLayout(layout)

    def detect_cameras(self):
        self.camera_selector.clear()
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            self.camera_selector.addItem(f"Camera {index}", index)
            cap.release()
            index += 1

        if self.camera_selector.count() > 0:
            self.change_camera(0)

    def change_camera(self, index):
        if self.video_capture:
            self.timer.stop()
            self.video_capture.release()

        camera_index = self.camera_selector.itemData(index)
        self.video_capture = cv2.VideoCapture(camera_index)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            processed_frame = self.apply_transformations(frame)
            self.current_frame = processed_frame
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def apply_transformations(self, frame):
        """Apply orientation adjustments (flip horizontally/vertically)."""
        flip_h = self.flip_horizontal_checkbox.isChecked()
        flip_v = self.flip_vertical_checkbox.isChecked()

        if flip_h and flip_v:
            # Flip both horizontally and vertically
            return cv2.flip(frame, -1)
        elif flip_h:
            return cv2.flip(frame, 1)
        elif flip_v:
            return cv2.flip(frame, 0)
        return frame

    def preprocess_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def capture_and_ocr(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Error", "No frame captured.")
            return

        processed = self.preprocess_image(self.current_frame)
        pil_image = Image.fromarray(processed)

        text = pytesseract.image_to_string(pil_image)

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if len(lines) == 0:
            QMessageBox.information(self, "No Text Found", "OCR did not detect text.")
            return

        # Limit to first 5 lines
        lines = lines[:5]

        self.extracted_data.append(lines)

        # Automatically export to CSV on each successful capture
        self.export_to_csv()

        QMessageBox.information(
            self,
            "OCR Success",
            f"Captured {len(lines)} lines.\n\n" + "\n".join(lines)
        )

    def browse_save_path(self):
        """Open file dialog to choose where to save or which file to append to."""
        start_path = self.path_edit.text().strip() or ""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save or select CSV file",
            start_path or "",
            "CSV Files (*.csv)"
        )
        if file_path:
            self.path_edit.setText(file_path)

    def export_to_csv(self):
        if not self.extracted_data:
            QMessageBox.warning(self, "No Data", "No OCR data to export.")
            return

        file_path = self.path_edit.text().strip()
        if not file_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save CSV",
                "",
                "CSV Files (*.csv)"
            )
            if file_path:
                self.path_edit.setText(file_path)

        if file_path:
            try:
                append_mode = self.append_checkbox.isChecked()
                mode = 'a' if append_mode else 'w'
                with open(file_path, mode=mode, newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    for entry in self.extracted_data:
                        writer.writerow(entry)

                QMessageBox.information(self, "Success", "Data exported successfully!")
                self.extracted_data.clear()

            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OCRScanner()
    window.show()
    sys.exit(app.exec())
