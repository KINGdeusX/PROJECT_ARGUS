import sys
import platform
import cv2
import csv
import pytesseract
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox,
    QFileDialog, QMessageBox, QComboBox,
    QLineEdit, QCheckBox, QDoubleSpinBox, QSlider,
    QSizePolicy
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
        self.setMinimumSize(640, 480)

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
        # Camera property definitions: (label, OpenCV property id, settings key) or + (min, max) for range
        self.camera_props = [
            ("Brightness", cv2.CAP_PROP_BRIGHTNESS, "brightness", 0, 255),
            ("Contrast", cv2.CAP_PROP_CONTRAST, "contrast", 0, 255),
            ("Saturation", cv2.CAP_PROP_SATURATION, "saturation", 0, 255),
            ("Gamma", cv2.CAP_PROP_GAMMA, "gamma", 0, 500),
            ("Exposure", cv2.CAP_PROP_EXPOSURE, "exposure", -13, 0),
            ("Gain", cv2.CAP_PROP_GAIN, "gain", 0, 255),
            ("Sharpness", getattr(cv2, "CAP_PROP_SHARPNESS", None), "sharpness", 0, 255),
            ("Focus", getattr(cv2, "CAP_PROP_FOCUS", None), "focus", 0, 255),
            ("White balance (K)", getattr(cv2, "CAP_PROP_WB_TEMPERATURE", None), "wb_temperature", 2000, 10000),
        ]
        # Drop any entry whose prop_id is None (unsupported in this OpenCV build)
        self.camera_props = [p for p in self.camera_props if p[1] is not None]
        # Keys that use a slider instead of a spinbox
        self.camera_slider_keys = {
            "brightness",
            "contrast",
            "saturation",
            "gamma",
            "exposure",
            "gain",
            "sharpness",
            "focus",
            "wb_temperature",
        }
        # Default values for reset (key -> value); sliders use int, spinboxes use float
        self.camera_defaults = {
            "brightness": 128.0,
            "contrast": 128,
            "saturation": 64.0,
            "gamma": 100.0,
            "exposure": -6,
            "gain": 0,
            "sharpness": 0,
            "focus": 0,
            "wb_temperature": 5000,
            "auto_exposure": 1.0,
            "auto_wb": 1.0,
            "auto_focus": 1.0,
            "lowlight_comp": 1.0,
        }
        # Common resolution presets (Logitech BRIO 100 and typical webcams)
        self.resolutions = [
            ("1920 x 1080 (Full HD)", 1920, 1080),
            ("1280 x 720 (HD)", 1280, 720),
            ("960 x 540", 960, 540),
            ("640 x 480", 640, 480),
            ("320 x 240", 320, 240),
        ]
        # FPS presets (camera-supported; timer is synced to this)
        self.fps_presets = [
            ("15 fps", 15),
            ("24 fps", 24),
            ("30 fps", 30),
            ("60 fps", 60),
        ]
        # key -> spinbox or slider
        self.camera_controls = {}
        # key -> value label (for slider rows, shows current value)
        self.camera_value_labels = {}
        # Optional toggle properties, only added if OpenCV exposes them
        self.camera_toggle_props = []
        auto_exposure_id = getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", None)
        if auto_exposure_id is not None:
            self.camera_toggle_props.append(
                ("Auto exposure", auto_exposure_id, "auto_exposure")
            )
        auto_wb_id = getattr(cv2, "CAP_PROP_AUTO_WB", None)
        if auto_wb_id is not None:
            self.camera_toggle_props.append(
                ("Auto white balance", auto_wb_id, "auto_wb")
            )
        autofocus_id = getattr(cv2, "CAP_PROP_AUTOFOCUS", None)
        if autofocus_id is not None:
            self.camera_toggle_props.append(
                ("Auto focus", autofocus_id, "auto_focus")
            )
        lowlight_id = getattr(cv2, "CAP_PROP_BACKLIGHT", None)
        if lowlight_id is not None:
            self.camera_toggle_props.append(
                ("Low-light compensation", lowlight_id, "lowlight_comp")
            )
        # key -> checkbox (toggle controls)
        self.camera_toggle_controls = {}
        self.settings = QSettings("ProjectArgus", "ClaimsScanner")

        self.init_ui()
        self.detect_cameras()
        self.load_settings()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # --- Right panel: live camera feed + capture button ---
        self.video_label = VideoLabel(self)
        self.video_label.setText("Camera Feed")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.capture_button = QPushButton("Capture & OCR")
        self.capture_button.clicked.connect(self.capture_and_ocr)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.video_label, stretch=1)
        right_layout.addWidget(self.capture_button, stretch=0)
        right_layout.setAlignment(self.capture_button, Qt.AlignmentFlag.AlignLeft)

        # --- Left panel: all controls/settings ---
        left_layout = QVBoxLayout()

        # Camera selector
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self.camera_selector = QComboBox()
        self.camera_selector.currentIndexChanged.connect(self.change_camera)
        cam_row.addWidget(self.camera_selector)
        left_layout.addLayout(cam_row)

        # Resolution selector
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        self.resolution_selector = QComboBox()
        for label, w, h in self.resolutions:
            self.resolution_selector.addItem(label, (w, h))
        self.resolution_selector.currentIndexChanged.connect(self.change_resolution)
        res_row.addWidget(self.resolution_selector)
        left_layout.addLayout(res_row)

        # FPS selector
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self.fps_selector = QComboBox()
        for label, fps_val in self.fps_presets:
            self.fps_selector.addItem(label, fps_val)
        self.fps_selector.currentIndexChanged.connect(self.change_fps)
        fps_row.addWidget(self.fps_selector)
        left_layout.addLayout(fps_row)

        # Region controls
        region_layout = QHBoxLayout()
        region_layout.addWidget(QLabel("Region: drag on feed"))
        self.clear_region_btn = QPushButton("Clear region")
        self.clear_region_btn.clicked.connect(self.video_label.clear_selection)
        region_layout.addWidget(self.clear_region_btn)
        left_layout.addLayout(region_layout)

        # Camera settings controls (talking directly to the driver via OpenCV)
        cam_group = QGroupBox("Camera settings")
        cam_layout = QGridLayout()
        # Numeric controls (all sliders now: brightness, contrast, gain, etc.)
        for row, prop_tuple in enumerate(self.camera_props):
            label_text, prop_id, key = prop_tuple[0], prop_tuple[1], prop_tuple[2]
            lbl = QLabel(label_text)
            if key in self.camera_slider_keys and len(prop_tuple) >= 5:
                low, high = int(prop_tuple[3]), int(prop_tuple[4])
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(low, high)
                slider.setSingleStep(max(1, (high - low) // 100))
                value_label = QLabel(str(low))
                value_label.setMinimumWidth(48)

                def make_slider_cb(pid, skey, val_lbl):
                    def on_change(val):
                        self.set_camera_property(pid, skey, float(val))
                        val_lbl.setText(str(val))

                    return on_change

                slider.valueChanged.connect(make_slider_cb(prop_id, key, value_label))
                cam_layout.addWidget(lbl, row, 0)
                cam_layout.addWidget(slider, row, 1)
                cam_layout.addWidget(value_label, row, 2)
                self.camera_controls[key] = slider
                self.camera_value_labels[key] = value_label
            else:
                spin = QDoubleSpinBox()
                if len(prop_tuple) >= 5:
                    spin.setRange(float(prop_tuple[3]), float(prop_tuple[4]))
                else:
                    spin.setRange(-10000.0, 10000.0)
                spin.setDecimals(2)
                spin.setSingleStep(1.0)
                spin.valueChanged.connect(
                    lambda val, pid=prop_id, skey=key: self.set_camera_property(
                        pid, skey, val
                    )
                )
                cam_layout.addWidget(lbl, row, 0)
                cam_layout.addWidget(spin, row, 1)
                self.camera_controls[key] = spin

        # Toggle controls (auto exposure / auto white balance / auto focus / low-light)
        toggle_row_start = len(self.camera_props)
        for idx, (label_text, prop_id, key) in enumerate(self.camera_toggle_props):
            if prop_id is None:
                continue
            chk = QCheckBox(label_text)
            chk.stateChanged.connect(
                lambda state, pid=prop_id, skey=key: self.set_camera_toggle_property(
                    pid, skey, state == Qt.CheckState.Checked
                )
            )
            cam_layout.addWidget(chk, toggle_row_start + idx, 0, 1, 3)
            self.camera_toggle_controls[key] = chk

        hint_row = toggle_row_start + len(self.camera_toggle_props)
        hint = QLabel(
            "Tip: Turn off \"Auto focus\" or \"Auto white balance\" "
            "to use the manual Focus and White balance (K) values above."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        cam_layout.addWidget(hint, hint_row, 0, 1, 3)

        reset_btn = QPushButton("Reset camera settings to defaults")
        reset_btn.clicked.connect(self.reset_camera_settings)
        cam_layout.addWidget(reset_btn, hint_row + 1, 0, 1, 3)

        cam_group.setLayout(cam_layout)
        left_layout.addWidget(cam_group)

        # Flip controls for camera orientation
        transform_layout = QHBoxLayout()
        self.flip_horizontal_checkbox = QCheckBox("Flip horizontally")
        self.flip_vertical_checkbox = QCheckBox("Flip vertically")
        transform_layout.addWidget(self.flip_horizontal_checkbox)
        transform_layout.addWidget(self.flip_vertical_checkbox)
        left_layout.addLayout(transform_layout)

        # File path row: input + browse
        path_layout = QHBoxLayout()
        self.path_label = QLabel("Save to:")
        path_layout.addWidget(self.path_label)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            "Select a file to save or append CSV data..."
        )
        path_layout.addWidget(self.path_edit)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_save_path)
        path_layout.addWidget(self.browse_button)
        left_layout.addLayout(path_layout)

        self.append_checkbox = QCheckBox("Append to existing file")
        self.append_checkbox.setChecked(True)
        left_layout.addWidget(self.append_checkbox)

        left_layout.addStretch(1)

        # Add panels to main layout (settings on the left, feed on the right)
        main_layout.addLayout(left_layout, stretch=0)
        main_layout.addLayout(right_layout, stretch=1)

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

    def load_camera_properties_for_current(self):
        """Load per-camera driver properties (exposure, contrast, etc.) into UI and device."""
        if not self.video_capture or not self.video_capture.isOpened():
            return
        camera_index = self.camera_selector.currentData()
        if camera_index is None:
            return
        # Numeric properties
        for prop_tuple in self.camera_props:
            _, prop_id, key = prop_tuple[0], prop_tuple[1], prop_tuple[2]
            spin = self.camera_controls.get(key)
            if spin is None:
                continue
            setting_key = f"camera/{camera_index}/{key}"
            saved = self.settings.value(setting_key, None)
            if saved is not None:
                try:
                    val = float(saved)
                except (TypeError, ValueError):
                    val = self.video_capture.get(prop_id)
                # Apply saved value to device
                if val is not None:
                    self.video_capture.set(prop_id, float(val))
            else:
                val = self.video_capture.get(prop_id)
            if val is None:
                continue
            ctrl = self.camera_controls.get(key)
            if ctrl is None:
                continue
            ctrl.blockSignals(True)
            if key in self.camera_slider_keys:
                ival = int(round(val))
                ctrl.setValue(ival)
                vlbl = self.camera_value_labels.get(key)
                if vlbl is not None:
                    vlbl.setText(str(ival))
            else:
                ctrl.setValue(val)
            ctrl.blockSignals(False)

        # Toggle properties (auto exposure / WB / focus)
        for _, prop_id, key in self.camera_toggle_props:
            chk = self.camera_toggle_controls.get(key)
            if chk is None or prop_id is None:
                continue
            setting_key = f"camera/{camera_index}/{key}"
            saved = self.settings.value(setting_key, None)
            if saved is not None:
                try:
                    val = float(saved)
                except (TypeError, ValueError):
                    val = self.video_capture.get(prop_id)
                if val is not None:
                    self.video_capture.set(prop_id, float(val))
            else:
                val = self.video_capture.get(prop_id)
            if val is None:
                continue
            checked = bool(val >= 0.5)
            chk.blockSignals(True)
            chk.setChecked(checked)
            chk.blockSignals(False)

    def load_resolution_for_current(self):
        """Apply saved resolution for the currently selected camera (and update dropdown)."""
        if self.camera_selector.count() == 0 or self.resolution_selector.count() == 0:
            return
        camera_index = self.camera_selector.currentData()
        if camera_index is None:
            return
        saved_index = self.settings.value(
            f"camera/{camera_index}/resolutionIndex", 0, type=int
        )
        if saved_index < 0 or saved_index >= self.resolution_selector.count():
            saved_index = 0
        self.resolution_selector.blockSignals(True)
        self.resolution_selector.setCurrentIndex(saved_index)
        self.resolution_selector.blockSignals(False)
        self.change_resolution(saved_index)

    def load_fps_for_current(self):
        """Apply saved FPS for the current camera and sync timer."""
        if self.camera_selector.count() == 0 or self.fps_selector.count() == 0:
            return
        camera_index = self.camera_selector.currentData()
        if camera_index is None:
            return
        saved_index = self.settings.value(
            f"camera/{camera_index}/fpsIndex", 2, type=int
        )  # default 30 fps (index 2)
        if saved_index < 0 or saved_index >= self.fps_selector.count():
            saved_index = 2
        self.fps_selector.blockSignals(True)
        self.fps_selector.setCurrentIndex(saved_index)
        self.fps_selector.blockSignals(False)
        self._apply_fps_and_timer(self.fps_selector.currentData())

    def _apply_fps_and_timer(self, fps_value):
        """Set camera FPS and timer interval from FPS value."""
        if fps_value is None or fps_value <= 0:
            fps_value = 30
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.set(cv2.CAP_PROP_FPS, int(fps_value))
        interval_ms = max(5, min(100, int(1000 / fps_value)))
        self.timer.setInterval(interval_ms)

    def change_fps(self, index):
        """Change capture FPS and timer interval."""
        if index < 0 or self.fps_selector.count() == 0:
            return
        fps_value = self.fps_selector.itemData(index)
        if fps_value is None:
            return
        self._apply_fps_and_timer(fps_value)
        camera_index = self.camera_selector.currentData()
        if camera_index is not None:
            self.settings.setValue(f"camera/{camera_index}/fpsIndex", int(index))
            self.settings.sync()

    def set_camera_property(self, prop_id, key, value):
        """Apply a camera driver property via OpenCV and persist it per camera."""
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.set(prop_id, float(value))
        camera_index = self.camera_selector.currentData()
        if camera_index is not None:
            setting_key = f"camera/{camera_index}/{key}"
            self.settings.setValue(setting_key, float(value))
            self.settings.sync()

    def set_camera_toggle_property(self, prop_id, key, enabled):
        """Apply a boolean-ish camera property and persist it per camera."""
        val = 1.0 if enabled else 0.0
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.set(prop_id, val)
        camera_index = self.camera_selector.currentData()
        if camera_index is not None:
            setting_key = f"camera/{camera_index}/{key}"
            self.settings.setValue(setting_key, val)
            self.settings.sync()

    def change_resolution(self, index):
        """Change capture resolution based on dropdown selection."""
        if index < 0 or self.resolution_selector.count() == 0:
            return
        data = self.resolution_selector.itemData(index)
        if not data:
            return
        width, height = data
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        camera_index = self.camera_selector.currentData()
        if camera_index is not None:
            self.settings.setValue(
                f"camera/{camera_index}/resolutionIndex", int(index)
            )
            self.settings.sync()

    def reset_camera_settings(self):
        """Reset all camera settings to defaults and apply to device."""
        if not self.video_capture or not self.video_capture.isOpened():
            QMessageBox.information(
                self, "Reset",
                "No camera open. Select a camera first."
            )
            return
        camera_index = self.camera_selector.currentData()
        if camera_index is None:
            return
        for prop_tuple in self.camera_props:
            _, prop_id, key = prop_tuple[0], prop_tuple[1], prop_tuple[2]
            default = self.camera_defaults.get(key)
            if default is None:
                continue
            val = float(default) if key not in self.camera_slider_keys else int(default)
            self.video_capture.set(prop_id, val)
            ctrl = self.camera_controls.get(key)
            if ctrl is not None:
                ctrl.blockSignals(True)
                if key in self.camera_slider_keys:
                    ctrl.setValue(int(round(val)))
                    vlbl = self.camera_value_labels.get(key)
                    if vlbl is not None:
                        vlbl.setText(str(int(round(val))))
                else:
                    ctrl.setValue(val)
                ctrl.blockSignals(False)
            self.settings.setValue(f"camera/{camera_index}/{key}", val)
        for _, prop_id, key in self.camera_toggle_props:
            if prop_id is None:
                continue
            default = self.camera_defaults.get(key, 1.0)
            val = float(default)
            self.video_capture.set(prop_id, val)
            chk = self.camera_toggle_controls.get(key)
            if chk is not None:
                chk.blockSignals(True)
                chk.setChecked(bool(val >= 0.5))
                chk.blockSignals(False)
            self.settings.setValue(f"camera/{camera_index}/{key}", val)
        self.settings.sync()
        QMessageBox.information(self, "Reset", "Camera settings reset to defaults.")

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
        # Apply saved resolution, FPS, and sync UI controls with this camera's driver settings
        self.load_resolution_for_current()
        self.load_fps_for_current()
        self.load_camera_properties_for_current()
        self.timer.start()

    def update_frame(self):
        if self.video_capture is None or not self.video_capture.isOpened():
            return
        # Drop stale frames so we always show the latest (reduces lag and stutter)
        for _ in range(4):
            if not self.video_capture.grab():
                break
        ret, frame = self.video_capture.retrieve()
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
