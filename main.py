import sys
import platform
import cv2
import csv
import pytesseract
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QComboBox,
    QLineEdit, QCheckBox
)
from PyQt6.QtCore import QTimer, Qt, QSettings, QRect, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent
from PIL import Image
import numpy as np

# Uncomment and adjust path if needed (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Max size for displayed video (selection is drawn on this)
DISPLAY_MAX_WIDTH = 800
DISPLAY_MAX_HEIGHT = 600

# Prefer DirectShow on Windows to avoid MSMF "can't grab frame" errors
def open_camera(index):
    """Open VideoCapture with backend that avoids MSMF grab errors on Windows."""
    if platform.system() == "Windows" and getattr(cv2, "CAP_DSHOW", None) is not None:
        cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(int(index))
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class VideoLabel(QLabel):
    """Label that displays the camera frame and allows drawing a capture region."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(False)
        self._frame_shape = (0, 0)  # (height, width) of current frame
        self._display_size = (0, 0)  # (width, height) of scaled pixmap
        self._selection = None  # (x, y, w, h) in frame coords
        self._selection_normalized = None  # (nx, ny, nw, nh) 0-1 for persistence
        self._drawing = False
        self._start_pos = None
        self._end_pos = None
        self.setMinimumSize(320, 240)

    def _image_rect(self):
        """Widget rect where the image is drawn (centered)."""
        if self._display_size[0] <= 0 or self._display_size[1] <= 0:
            return QRect(0, 0, self.width(), self.height())
        x = (self.width() - self._display_size[0]) // 2
        y = (self.height() - self._display_size[1]) // 2
        return QRect(int(x), int(y), self._display_size[0], self._display_size[1])

    def _widget_to_frame_rect(self, x_w, y_w, w_w, h_w):
        """Convert widget rect to frame coords (x, y, w, h)."""
        ir = self._image_rect()
        fh, fw = self._frame_shape
        if fw <= 0 or fh <= 0 or ir.width() <= 0 or ir.height() <= 0:
            return None
        # Clamp to image area
        x1 = max(0, min(x_w - ir.x(), ir.width()))
        y1 = max(0, min(y_w - ir.y(), ir.height()))
        x2 = max(0, min(x_w - ir.x() + w_w, ir.width()))
        y2 = max(0, min(y_w - ir.y() + h_w, ir.height()))
        if x2 <= x1 or y2 <= y1:
            return None
        x_f = int(x1 * fw / ir.width())
        y_f = int(y1 * fh / ir.height())
        w_f = int((x2 - x1) * fw / ir.width())
        h_f = int((y2 - y1) * fh / ir.height())
        if w_f <= 0 or h_f <= 0:
            return None
        return (x_f, y_f, w_f, h_f)

    def _frame_to_widget_rect(self, x_f, y_f, w_f, h_f):
        """Convert frame rect to widget coords for drawing."""
        ir = self._image_rect()
        fh, fw = self._frame_shape
        if fw <= 0 or fh <= 0:
            return None
        x_w = ir.x() + int(x_f * ir.width() / fw)
        y_w = ir.y() + int(y_f * ir.height() / fh)
        w_w = int(w_f * ir.width() / fw)
        h_w = int(h_f * ir.height() / fh)
        return QRect(x_w, y_w, w_w, h_w)

    def update_frame(self, frame):
        """Set the display from an OpenCV BGR frame. Scales to fit and stores shape."""
        if frame is None or frame.size == 0:
            return
        self._frame_shape = (frame.shape[0], frame.shape[1])
        h, w = self._frame_shape
        scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w < 1 or new_h < 1:
            return
        scaled = cv2.resize(frame, (new_w, new_h))
        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        bytes_per_line = rgb.shape[1] * 3
        qt_image = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0],
            bytes_per_line, QImage.Format.Format_RGB888
        )
        self._display_size = (qt_image.width(), qt_image.height())
        self.setPixmap(QPixmap.fromImage(qt_image.copy()))
        # Restore selection from normalized if we had one and no frame coords yet
        if self._selection_normalized and self._selection is None:
            nx, ny, nw, nh = self._selection_normalized
            self._selection = (
                int(nx * w), int(ny * h), int(nw * w), int(nh * h)
            )
        self.update()

    def get_selection(self):
        """Return (x, y, w, h) in frame coords or None for full frame."""
        return self._selection

    def set_selection_normalized(self, nx, ny, nw, nh):
        """Set selection from normalized 0-1 coords (e.g. from settings)."""
        if nx is None or ny is None or nw is None or nh is None:
            self._selection_normalized = None
            self._selection = None
            return
        self._selection_normalized = (
            max(0, min(1, nx)), max(0, min(1, ny)),
            max(0.01, min(1, nw)), max(0.01, min(1, nh))
        )
        fh, fw = self._frame_shape
        if fw > 0 and fh > 0:
            self._selection = (
                int(self._selection_normalized[0] * fw),
                int(self._selection_normalized[1] * fh),
                int(self._selection_normalized[2] * fw),
                int(self._selection_normalized[3] * fh),
            )
        self.update()

    def get_selection_normalized(self):
        """Return (nx, ny, nw, nh) 0-1 or None."""
        return self._selection_normalized

    def clear_selection(self):
        """Clear the capture region (use full frame)."""
        self._selection = None
        self._selection_normalized = None
        self._drawing = False
        self._start_pos = None
        self._end_pos = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            ir = self._image_rect()
            pos = event.position() if hasattr(event, 'position') else event.pos()
            x, y = int(pos.x()), int(pos.y())
            if ir.contains(x, y):
                self._drawing = True
                self._start_pos = (x, y)
                self._end_pos = (x, y)
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing and self._end_pos is not None:
            pos = event.position() if hasattr(event, 'position') else event.pos()
            self._end_pos = (int(pos.x()), int(pos.y()))
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            pos = event.position() if hasattr(event, 'position') else event.pos()
            self._end_pos = (int(pos.x()), int(pos.y()))
            x1, y1 = self._start_pos
            x2, y2 = self._end_pos
            x_w = min(x1, x2)
            y_w = min(y1, y2)
            w_w = max(abs(x2 - x1), 5)
            h_w = max(abs(y2 - y1), 5)
            rect = self._widget_to_frame_rect(x_w, y_w, w_w, h_w)
            if rect:
                self._selection = rect
                fh, fw = self._frame_shape
                if fw > 0 and fh > 0:
                    self._selection_normalized = (
                        self._selection[0] / fw, self._selection[1] / fh,
                        self._selection[2] / fw, self._selection[3] / fh,
                    )
            self._start_pos = None
            self._end_pos = None
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._drawing and self._start_pos and self._end_pos:
            x1, y1 = self._start_pos
            x2, y2 = self._end_pos
            r = QRect(
                min(x1, x2), min(y1, y2),
                max(abs(x2 - x1), 2), max(abs(y2 - y1), 2)
            )
            painter.drawRect(r)
        elif self._selection:
            wr = self._frame_to_widget_rect(
                self._selection[0], self._selection[1],
                self._selection[2], self._selection[3]
            )
            if wr:
                painter.drawRect(wr)
        painter.end()


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
        self.settings = QSettings("ProjectArgus", "ClaimsScanner")

        self.init_ui()
        self.detect_cameras()
        self.load_settings()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.camera_selector = QComboBox()
        self.camera_selector.currentIndexChanged.connect(self.change_camera)
        layout.addWidget(self.camera_selector)

        self.video_label = VideoLabel(self)
        self.video_label.setText("Camera Feed")
        layout.addWidget(self.video_label)

        region_layout = QHBoxLayout()
        region_layout.addWidget(QLabel("Capture region: drag on the feed to select. OCR uses only that area."))
        self.clear_region_btn = QPushButton("Clear region (use full frame)")
        self.clear_region_btn.clicked.connect(self.video_label.clear_selection)
        region_layout.addWidget(self.clear_region_btn)
        layout.addLayout(region_layout)

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
            cap = open_camera(index)
            if not cap.isOpened() or not cap.read()[0]:
                cap.release()
                break
            self.camera_selector.addItem(f"Camera {index}", index)
            cap.release()
            index += 1

        if self.camera_selector.count() > 0:
            # Default to first camera; load_settings() may override with saved index
            self.change_camera(0)

    def load_settings(self):
        """Restore saved camera index, orientation, and file path."""
        saved_camera = self.settings.value("cameraIndex", 0, type=int)
        for i in range(self.camera_selector.count()):
            if self.camera_selector.itemData(i) == saved_camera:
                self.camera_selector.blockSignals(True)
                self.camera_selector.setCurrentIndex(i)
                self.camera_selector.blockSignals(False)
                self.change_camera(i)
                break

        self.flip_horizontal_checkbox.setChecked(
            self.settings.value("flipHorizontal", False, type=bool)
        )
        self.flip_vertical_checkbox.setChecked(
            self.settings.value("flipVertical", False, type=bool)
        )
        self.path_edit.setText(self.settings.value("saveFilePath", "", type=str))
        self.append_checkbox.setChecked(
            self.settings.value("appendToExisting", True, type=bool)
        )
        # Restore capture region if saved
        snx = self.settings.value("selectionNormX", None)
        sny = self.settings.value("selectionNormY", None)
        snw = self.settings.value("selectionNormW", None)
        snh = self.settings.value("selectionNormH", None)
        if snx is not None and sny is not None and snw is not None and snh is not None:
            try:
                self.video_label.set_selection_normalized(
                    float(snx), float(sny), float(snw), float(snh)
                )
            except (TypeError, ValueError):
                pass

    def save_settings(self):
        """Persist camera index, orientation, and file path."""
        if self.camera_selector.count() > 0:
            self.settings.setValue(
                "cameraIndex",
                self.camera_selector.currentData(),
            )
        self.settings.setValue(
            "flipHorizontal",
            self.flip_horizontal_checkbox.isChecked(),
        )
        self.settings.setValue(
            "flipVertical",
            self.flip_vertical_checkbox.isChecked(),
        )
        self.settings.setValue("saveFilePath", self.path_edit.text().strip())
        self.settings.setValue(
            "appendToExisting",
            self.append_checkbox.isChecked(),
        )
        norm = self.video_label.get_selection_normalized()
        if norm is not None:
            self.settings.setValue("selectionNormX", norm[0])
            self.settings.setValue("selectionNormY", norm[1])
            self.settings.setValue("selectionNormW", norm[2])
            self.settings.setValue("selectionNormH", norm[3])
        else:
            self.settings.remove("selectionNormX")
            self.settings.remove("selectionNormY")
            self.settings.remove("selectionNormW")
            self.settings.remove("selectionNormH")
        self.settings.sync()

    def change_camera(self, index):
        if self.video_capture:
            self.timer.stop()
            self.video_capture.release()
            self.video_capture = None

        camera_index = self.camera_selector.itemData(index)
        self.video_capture = open_camera(camera_index)
        if not self.video_capture.isOpened():
            self.video_capture = None
            QMessageBox.warning(
                self, "Camera error",
                "Could not open the selected camera. Try another camera index."
            )
            return
        self.timer.start(30)

    def update_frame(self):
        if self.video_capture is None or not self.video_capture.isOpened():
            return
        ret, frame = self.video_capture.read()
        if not ret or frame is None:
            return
        processed_frame = self.apply_transformations(frame)
        self.current_frame = processed_frame
        self.video_label.update_frame(processed_frame)

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

        frame = self.current_frame
        sel = self.video_label.get_selection()
        if sel is not None:
            x, y, w, h = sel
            frame = frame[y : y + h, x : x + w]

        processed = self.preprocess_image(frame)
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

    def closeEvent(self, event):
        """Save config when closing the window."""
        self.save_settings()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OCRScanner()
    window.show()
    sys.exit(app.exec())
