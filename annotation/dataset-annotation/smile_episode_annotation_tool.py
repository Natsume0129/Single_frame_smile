from __future__ import annotations

import sys
from pathlib import Path

import cv2

try:
    from PySide6.QtCore import QElapsedTimer, QEvent, Qt, QTimer
    from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QSplitter,
        QStyle,
        QStyleOptionSlider,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print(
        "PySide6 is required to run this desktop tool. "
        "Install dependencies with: pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise

from annotation_store import (
    AnnotationStore,
    EpisodeDraft,
    MAIN_LABELS,
    SYMMETRY_VALUES,
    USABLE_VALUES,
    VISIBLE_QUALITY_VALUES,
    default_usable_for_training,
    label_requires_peak,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = APP_DIR / "annotations.csv"
VIDEO_FILTER = "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)"


class FrameSlider(QSlider):
    COLORS = {
        "start": QColor(40, 170, 95),
        "peak": QColor(220, 60, 70),
        "end": QColor(55, 120, 220),
    }

    def __init__(self, orientation: Qt.Orientation, parent: QWidget | None = None) -> None:
        super().__init__(orientation, parent)
        self._markers: dict[str, int | None] = {"start": None, "peak": None, "end": None}
        self.setMinimumHeight(34)

    def set_marker(self, name: str, frame_index: int | None) -> None:
        self._markers[name] = frame_index
        self.update()

    def clear_markers(self) -> None:
        for name in self._markers:
            self._markers[name] = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        super().paintEvent(event)
        if self.maximum() <= self.minimum():
            return

        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        span = max(1, self.width() - handle.width())
        min_value = self.minimum()
        max_value = self.maximum()
        top = max(0, groove.top() - 8)
        bottom = min(self.height() - 1, groove.bottom() + 8)

        for name, frame_index in self._markers.items():
            if frame_index is None:
                continue
            ratio = (frame_index - min_value) / max(1, max_value - min_value)
            x = int(handle.width() / 2 + ratio * span)
            painter.setPen(QPen(self.COLORS.get(name, QColor("white")), 3))
            painter.drawLine(x, top, x, bottom)


class VideoLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(960, 540)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #111; color: #bbb;")
        self.setText("Load a video to begin")

    def set_frame(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._rescale()

    def clear_frame(self) -> None:
        self._source_pixmap = None
        self.clear()
        self.setText("Load a video to begin")

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class SmileEpisodeAnnotationWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Smile Episode Annotation Tool")
        self.resize(1680, 980)

        self.store = AnnotationStore(DEFAULT_CSV_PATH)
        self.capture: cv2.VideoCapture | None = None
        self.video_path: str | None = None
        self.fps = 0.0
        self.total_frames = 0
        self.current_frame = 0
        self.current_marks: dict[str, int | None] = {"start": None, "peak": None, "end": None}
        self.current_episode_rows: list[dict[str, str]] = []
        self.loaded_episode_id: str | None = None
        self._updating_slider = False
        self.playback_rate = 1.0
        self.playback_start_frame = 0
        self.playback_stop_frame: int | None = None
        self.play_clock = QElapsedTimer()

        self.play_timer = QTimer(self)
        self.play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.play_timer.timeout.connect(self._advance_playback)

        self._build_ui()
        self._install_shortcuts()
        QApplication.instance().installEventFilter(self)
        self._refresh_info()
        self._refresh_episode_table()

    def closeEvent(self, event) -> None:  # noqa: ANN001
        QApplication.instance().removeEventFilter(self)
        if self.capture is not None:
            self.capture.release()
        super().closeEvent(event)

    def eventFilter(self, obj, event) -> bool:  # noqa: ANN001
        if event.type() == QEvent.Type.KeyPress and self._handle_shortcut_key(event):
            return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if self._handle_shortcut_key(event):
            return
        super().keyPressEvent(event)

    def _handle_shortcut_key(self, event) -> bool:  # noqa: ANN001
        focus = QApplication.focusWidget()
        combo_popup_open = isinstance(focus, QComboBox) and focus.view().isVisible()
        editing = isinstance(focus, (QLineEdit, QTextEdit, QAbstractSpinBox)) or combo_popup_open
        if editing:
            return False

        key = event.key()
        if key == Qt.Key.Key_Space:
            if event.isAutoRepeat():
                return True
            self.toggle_play_pause()
        elif key == Qt.Key.Key_Left:
            self.jump_frames(-1)
        elif key == Qt.Key.Key_Right:
            self.jump_frames(1)
        elif key == Qt.Key.Key_A:
            self.jump_frames(-5)
        elif key == Qt.Key.Key_D:
            self.jump_frames(5)
        elif key == Qt.Key.Key_J:
            self.jump_seconds(-1)
        elif key == Qt.Key.Key_L:
            self.jump_seconds(1)
        elif key == Qt.Key.Key_S:
            self.set_mark("start")
        elif key == Qt.Key.Key_P:
            self.set_mark("peak")
        elif key == Qt.Key.Key_E:
            self.set_mark("end")
        elif key == Qt.Key.Key_F11:
            self._toggle_fullscreen()
        elif key == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
        else:
            return False
        return True

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        self.setCentralWidget(central)

        top_bar = QHBoxLayout()
        self.open_button = self._make_button(
            "Open Video",
            self.open_video_dialog,
            QStyle.StandardPixmap.SP_DialogOpenButton,
        )
        self.next_video_button = self._make_button(
            "Next Video",
            self.open_video_dialog,
            QStyle.StandardPixmap.SP_FileDialogDetailedView,
        )
        self.csv_label = QLabel(f"CSV: {DEFAULT_CSV_PATH}")
        self.csv_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top_bar.addWidget(self.open_button)
        top_bar.addWidget(self.next_video_button)
        top_bar.addStretch(1)
        top_bar.addWidget(self.csv_label)
        root.addLayout(top_bar)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(content_splitter, stretch=1)

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.video_label = VideoLabel(self)
        left_layout.addWidget(self.video_label, stretch=1)

        self.slider = FrameSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        left_layout.addWidget(self.slider)

        playback_row = QHBoxLayout()
        self.play_button = self._make_button(
            "Play",
            self.toggle_play_pause,
            QStyle.StandardPixmap.SP_MediaPlay,
        )
        playback_row.addWidget(self.play_button)
        playback_row.addWidget(QLabel("Speed"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItem("1.0x", 1.0)
        self.speed_combo.addItem("0.5x", 0.5)
        self.speed_combo.currentIndexChanged.connect(self._set_playback_rate)
        playback_row.addWidget(self.speed_combo)
        playback_row.addWidget(
            self._make_button("Prev Frame", lambda: self.jump_frames(-1), QStyle.StandardPixmap.SP_MediaSkipBackward)
        )
        playback_row.addWidget(
            self._make_button("Next Frame", lambda: self.jump_frames(1), QStyle.StandardPixmap.SP_MediaSkipForward)
        )
        playback_row.addWidget(
            self._make_button("Back 5", lambda: self.jump_frames(-5), QStyle.StandardPixmap.SP_MediaSeekBackward)
        )
        playback_row.addWidget(
            self._make_button("Forward 5", lambda: self.jump_frames(5), QStyle.StandardPixmap.SP_MediaSeekForward)
        )
        playback_row.addWidget(
            self._make_button("Back 1s", lambda: self.jump_seconds(-1), QStyle.StandardPixmap.SP_MediaSeekBackward)
        )
        playback_row.addWidget(
            self._make_button("Forward 1s", lambda: self.jump_seconds(1), QStyle.StandardPixmap.SP_MediaSeekForward)
        )
        playback_row.addStretch(1)
        left_layout.addLayout(playback_row)

        right_panel = QWidget(self)
        right_panel.setMinimumWidth(420)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        info_group = QGroupBox("Video")
        info_grid = QGridLayout(info_group)
        self.filename_value = QLabel("-")
        self.path_value = QLabel("-")
        self.path_value.setWordWrap(True)
        self.path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.fps_value = QLabel("-")
        self.total_value = QLabel("-")
        self.current_frame_value = QLabel("-")
        self.current_time_value = QLabel("-")
        self._add_info_row(info_grid, 0, "Filename", self.filename_value)
        self._add_info_row(info_grid, 1, "Path", self.path_value)
        self._add_info_row(info_grid, 2, "FPS", self.fps_value)
        self._add_info_row(info_grid, 3, "Total frames", self.total_value)
        self._add_info_row(info_grid, 4, "Current frame", self.current_frame_value)
        self._add_info_row(info_grid, 5, "Current time", self.current_time_value)
        right_layout.addWidget(info_group)

        right_layout.addWidget(self._build_episode_group())
        right_layout.addWidget(self._build_attributes_group())

        action_row = QHBoxLayout()
        self.save_button = self._make_button(
            "Save Episode",
            self.save_episode,
            QStyle.StandardPixmap.SP_DialogSaveButton,
        )
        self.clear_button = self._make_button(
            "Clear Current Episode",
            self.clear_current_episode,
            QStyle.StandardPixmap.SP_DialogResetButton,
        )
        self.play_episode_button = self._make_button(
            "Play Selected Episode",
            self.play_selected_episode,
            QStyle.StandardPixmap.SP_MediaPlay,
        )
        self.delete_button = self._make_button(
            "Delete Selected Episode",
            self.delete_selected_episode,
            QStyle.StandardPixmap.SP_TrashIcon,
        )
        action_row.addWidget(self.save_button)
        action_row.addWidget(self.clear_button)
        action_row.addWidget(self.play_episode_button)
        action_row.addWidget(self.delete_button)
        action_row.addStretch(1)
        right_layout.addLayout(action_row)

        table_group = QGroupBox("Episode List for Current Video")
        table_layout = QVBoxLayout(table_group)
        self.episode_table = QTableWidget(0, 7)
        self.episode_table.setMinimumHeight(220)
        self.episode_table.setHorizontalHeaderLabels(
            ["episode_id", "start", "peak", "end", "label", "conf", "usable"]
        )
        self.episode_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.episode_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.episode_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.episode_table.cellClicked.connect(self._jump_to_table_episode)
        table_layout.addWidget(self.episode_table)
        right_layout.addWidget(table_group, stretch=1)

        right_scroll = QScrollArea(self)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(440)
        right_scroll.setWidget(right_panel)

        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(right_scroll)
        content_splitter.setStretchFactor(0, 4)
        content_splitter.setStretchFactor(1, 0)
        content_splitter.setSizes([1200, 440])

        self.statusBar().showMessage("Ready")

    def _build_episode_group(self) -> QGroupBox:
        group = QGroupBox("Episode Frames")
        grid = QGridLayout(group)

        self.start_frame_value = QLabel("-")
        self.peak_frame_value = QLabel("-")
        self.end_frame_value = QLabel("-")

        grid.addWidget(QLabel("Start frame"), 0, 0)
        grid.addWidget(self.start_frame_value, 0, 1)
        grid.addWidget(self._make_button("Set Start", lambda: self.set_mark("start")), 0, 2)
        grid.addWidget(self._make_button("Go", lambda: self.jump_to_mark("start")), 0, 3)

        grid.addWidget(QLabel("Peak frame"), 1, 0)
        grid.addWidget(self.peak_frame_value, 1, 1)
        self.set_peak_button = self._make_button("Set Peak", lambda: self.set_mark("peak"))
        grid.addWidget(self.set_peak_button, 1, 2)
        grid.addWidget(self._make_button("Go", lambda: self.jump_to_mark("peak")), 1, 3)
        grid.addWidget(self._make_button("Clear", self.clear_peak_mark), 1, 4)

        grid.addWidget(QLabel("End frame"), 2, 0)
        grid.addWidget(self.end_frame_value, 2, 1)
        grid.addWidget(self._make_button("Set End", lambda: self.set_mark("end")), 2, 2)
        grid.addWidget(self._make_button("Go", lambda: self.jump_to_mark("end")), 2, 3)

        self.person_id_edit = QLineEdit()
        self.person_id_edit.setPlaceholderText("Optional, e.g. P01")
        grid.addWidget(QLabel("Person ID"), 3, 0)
        grid.addWidget(self.person_id_edit, 3, 1, 1, 2)

        return group

    def _build_attributes_group(self) -> QGroupBox:
        group = QGroupBox("Episode Label")
        grid = QGridLayout(group)

        self.label_combo = QComboBox()
        self.label_combo.addItems(MAIN_LABELS)
        self.confidence_spin = self._spinbox(default=4)
        self.intensity_spin = self._spinbox(default=3)
        self.eye_spin = self._spinbox(default=3)
        self.mouth_spin = self._spinbox(default=3)
        self.cheek_spin = self._spinbox(default=3)
        self.symmetry_combo = QComboBox()
        self.symmetry_combo.addItems(SYMMETRY_VALUES)
        self.visible_quality_combo = QComboBox()
        self.visible_quality_combo.addItems(VISIBLE_QUALITY_VALUES)
        self.usable_combo = QComboBox()
        self.usable_combo.addItems(USABLE_VALUES)
        self.note_edit = QTextEdit()
        self.note_edit.setPlaceholderText("Optional note")
        self.note_edit.setMaximumHeight(72)

        fields = [
            ("Label", self.label_combo),
            ("Confidence", self.confidence_spin),
            ("Intensity", self.intensity_spin),
            ("Eye involvement", self.eye_spin),
            ("Mouth movement", self.mouth_spin),
            ("Cheek raise", self.cheek_spin),
            ("Symmetry", self.symmetry_combo),
            ("Visible quality", self.visible_quality_combo),
            ("Usable for training", self.usable_combo),
            ("Note", self.note_edit),
        ]
        for row, (label, widget) in enumerate(fields):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)

        self.label_combo.currentIndexChanged.connect(self._apply_default_usable)
        self.label_combo.currentIndexChanged.connect(self._refresh_mark_labels)
        self.confidence_spin.valueChanged.connect(self._apply_default_usable)
        self.visible_quality_combo.currentIndexChanged.connect(self._apply_default_usable)
        self._apply_default_usable()

        return group

    def _install_shortcuts(self) -> None:
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_episode)

    def _make_button(
        self,
        text: str,
        callback,
        icon: QStyle.StandardPixmap | None = None,  # noqa: ANN001
    ) -> QPushButton:
        button = QPushButton(text)
        if icon is not None:
            button.setIcon(self.style().standardIcon(icon))
        button.clicked.connect(callback)
        return button

    def _spinbox(self, default: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(1, 5)
        spinbox.setValue(default)
        return spinbox

    def _add_info_row(self, grid: QGridLayout, row: int, label: str, value_widget: QLabel) -> None:
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(value_widget, row, 1)

    def open_video_dialog(self) -> None:
        start_dir = str(Path(self.video_path).parent) if self.video_path else str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", start_dir, VIDEO_FILTER)
        if file_path:
            self.load_video(file_path)

    def load_video(self, file_path: str) -> None:
        self.pause()
        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            QMessageBox.critical(self, "Video Error", f"Could not open video:\n{file_path}")
            return

        raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            capture.release()
            QMessageBox.critical(self, "Video Error", "OpenCV could not read the video frame count.")
            return

        if self.capture is not None:
            self.capture.release()
        self.capture = capture
        self.video_path = str(Path(file_path).resolve())
        self.fps = raw_fps if raw_fps > 0 else 30.0
        self.total_frames = total_frames
        self.current_frame = 0
        self._update_play_timer_interval()

        self.slider.setRange(0, self.total_frames - 1)
        self.clear_current_episode(clear_note=False)
        self._show_frame(0)
        self._refresh_episode_table()
        self.statusBar().showMessage(f"Loaded {Path(file_path).name}")

    def toggle_play_pause(self) -> None:
        if self.capture is None:
            self._warn("Load a video before playback.")
            return
        if self.play_timer.isActive():
            self.pause()
            return
        if self.current_frame >= self.total_frames - 1:
            self._show_frame(0)
        if self.playback_stop_frame is not None and self.current_frame >= self.playback_stop_frame:
            self.playback_stop_frame = None
        self._reset_playback_clock()
        self.play_timer.start()
        self._update_play_button()

    def pause(self, clear_segment: bool = False) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
        if clear_segment:
            self.playback_stop_frame = None
        self._update_play_button()

    def next_frame(self) -> None:
        if self.capture is None:
            return
        if self.current_frame >= self.total_frames - 1:
            self.pause()
            return
        self._show_frame(self.current_frame + 1)

    def _advance_playback(self) -> None:
        if self.capture is None:
            return
        stop_frame = self.playback_stop_frame
        if stop_frame is None:
            stop_frame = self.total_frames - 1
        stop_frame = self._clamp_frame(stop_frame)

        if self.current_frame >= stop_frame:
            self.pause(clear_segment=True)
            return

        elapsed_frames = int((self.play_clock.elapsed() / 1000.0) * self.fps * self.playback_rate)
        target_frame = min(self._clamp_frame(self.playback_start_frame + elapsed_frames), stop_frame)
        if target_frame <= self.current_frame:
            return

        if target_frame == self.current_frame + 1:
            self._read_next_frame()
        else:
            self._show_frame(target_frame)

        if self.current_frame >= stop_frame:
            self.pause(clear_segment=True)

    def jump_frames(self, delta: int) -> None:
        if self.capture is None:
            self._warn("Load a video before seeking.")
            return
        self._seek_for_user(self._clamp_frame(self.current_frame + delta))

    def jump_seconds(self, seconds: int) -> None:
        frame_delta = int(round(seconds * self.fps))
        if frame_delta == 0:
            frame_delta = 1 if seconds > 0 else -1
        self.jump_frames(frame_delta)

    def set_mark(self, name: str) -> None:
        if self.capture is None:
            self._warn("Load a video before marking frames.")
            return
        self.current_marks[name] = self.current_frame
        self._refresh_mark_labels()
        self.slider.set_marker(name, self.current_frame)
        self.statusBar().showMessage(f"Set {name} frame to {self.current_frame}")

    def clear_peak_mark(self) -> None:
        self.current_marks["peak"] = None
        self.slider.set_marker("peak", None)
        self._refresh_mark_labels()
        self.statusBar().showMessage("Cleared peak frame")

    def jump_to_mark(self, name: str) -> None:
        if self.capture is None:
            self._warn("Load a video before jumping to a marked frame.")
            return
        frame_index = self.current_marks.get(name)
        if frame_index is None:
            self._warn(f"{name} frame is not set.")
            return
        self._seek_for_user(frame_index)
        self.statusBar().showMessage(f"Jumped to {name} frame {frame_index}")

    def clear_current_episode(self, clear_note: bool = True) -> None:
        self.current_marks = {"start": None, "peak": None, "end": None}
        self.loaded_episode_id = None
        self._refresh_mark_labels()
        if hasattr(self, "slider"):
            self.slider.clear_markers()
        if clear_note and hasattr(self, "note_edit"):
            self.note_edit.clear()
        self.statusBar().showMessage("Cleared current episode selection")

    def save_episode(self) -> None:
        if self.capture is None or self.video_path is None:
            self._warn("Load a video before saving an episode.")
            return

        if self.current_marks["start"] is None or self.current_marks["end"] is None:
            self._warn("Set start and end frames before saving.")
            return

        main_label = self.label_combo.currentText()
        if label_requires_peak(main_label) and self.current_marks["peak"] is None:
            self._warn("Set peak frame before saving a smile episode.")
            return
        peak_frame = (
            int(self.current_marks["peak"])
            if label_requires_peak(main_label) and self.current_marks["peak"] is not None
            else None
        )

        draft = EpisodeDraft(
            video_path=self.video_path,
            person_id=self.person_id_edit.text().strip(),
            start_frame=int(self.current_marks["start"]),
            peak_frame=peak_frame,
            end_frame=int(self.current_marks["end"]),
            fps=self.fps,
            main_label=main_label,
            confidence=self.confidence_spin.value(),
            intensity=self.intensity_spin.value(),
            eye_involvement=self.eye_spin.value(),
            mouth_movement=self.mouth_spin.value(),
            cheek_raise=self.cheek_spin.value(),
            symmetry=self.symmetry_combo.currentText(),
            visible_quality=self.visible_quality_combo.currentText(),
            usable_for_training=self.usable_combo.currentText(),
            note=self.note_edit.toPlainText().strip(),
        )

        if self.loaded_episode_id is not None:
            self._update_loaded_episode(draft)
        else:
            self._append_new_episode(draft)

    def _append_new_episode(self, draft: EpisodeDraft) -> None:
        try:
            row = self.store.append_episode(draft)
        except ValueError as exc:
            self._warn(str(exc))
            return

        next_start_frame = draft.end_frame
        self._refresh_episode_table()
        self._prepare_next_episode_start(next_start_frame)
        self.statusBar().showMessage(
            f"Saved {row['episode_id']} to {DEFAULT_CSV_PATH.name}; next start set to frame {next_start_frame}"
        )

    def _update_loaded_episode(self, draft: EpisodeDraft) -> None:
        episode_id = self.loaded_episode_id
        try:
            row = self.store.update_episode(episode_id, draft)
        except ValueError as exc:
            self._warn(str(exc))
            return

        if row is None:
            self._warn(f"Episode {episode_id} was not found in annotations.csv.")
            self.loaded_episode_id = None
            self._refresh_episode_table()
            return

        self._refresh_episode_table()
        self._select_episode_id(episode_id)
        self._load_episode_into_form(row)
        start_frame = self.current_marks.get("start")
        if start_frame is not None:
            self._seek_for_user(start_frame)
        self.statusBar().showMessage(f"Updated {episode_id} in {DEFAULT_CSV_PATH.name}")

    def delete_selected_episode(self) -> None:
        row_index = self._selected_episode_row()
        if row_index is None:
            self._warn("Select an episode row before deleting.")
            return

        episode = self.current_episode_rows[row_index]
        episode_id = episode.get("episode_id", "")
        if not episode_id:
            self._warn("Selected episode does not have an episode_id.")
            return

        answer = QMessageBox.question(
            self,
            "Delete Episode",
            f"Delete episode {episode_id} from annotations.csv?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        deleted = self.store.delete_episode(episode_id)
        if deleted is None:
            self._warn(f"Episode {episode_id} was not found in annotations.csv.")
            self._refresh_episode_table()
            return

        if self.loaded_episode_id == episode_id:
            self.clear_current_episode()
        self._refresh_episode_table()
        self.statusBar().showMessage(f"Deleted {episode_id}")

    def play_selected_episode(self) -> None:
        if self.capture is None:
            self._warn("Load a video before playing an episode.")
            return

        row_index = self._selected_episode_row()
        if row_index is None:
            self._warn("Select an episode row before playing it.")
            return

        episode = self.current_episode_rows[row_index]
        start_frame = _optional_int(episode.get("start_frame", ""))
        end_frame = _optional_int(episode.get("end_frame", ""))
        if start_frame is None or end_frame is None:
            self._warn("Selected episode does not have valid start/end frames.")
            return
        if start_frame >= end_frame:
            self._warn("Selected episode has an invalid frame range.")
            return

        self._load_episode_into_form(episode)
        self._play_frame_range(start_frame, end_frame)
        self.statusBar().showMessage(
            f"Playing {episode.get('episode_id', 'episode')} from frame {start_frame} to {end_frame}"
        )

    def _show_frame(self, frame_index: int) -> None:
        if self.capture is None:
            return

        frame_index = self._clamp_frame(frame_index)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.pause()
            self._warn(f"Could not read frame {frame_index}.")
            return

        self._display_frame(frame_index, frame)

    def _read_next_frame(self) -> None:
        if self.capture is None:
            return

        expected_frame = self._clamp_frame(self.current_frame + 1)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.pause()
            self._warn(f"Could not read frame {expected_frame}.")
            return

        actual_pos = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES) or expected_frame + 1)
        frame_index = self._clamp_frame(actual_pos - 1)
        self._display_frame(frame_index, frame)

    def _display_frame(self, frame_index: int, frame) -> None:  # noqa: ANN001
        self.current_frame = frame_index
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self.video_label.set_frame(QPixmap.fromImage(image))
        self._refresh_info()

        self._updating_slider = True
        self.slider.setValue(frame_index)
        self._updating_slider = False

    def _seek_for_user(self, frame_index: int) -> None:
        self._show_frame(frame_index)
        if self.play_timer.isActive():
            self._reset_playback_clock()

    def _play_frame_range(self, start_frame: int, end_frame: int) -> None:
        self.pause(clear_segment=True)
        start_frame = self._clamp_frame(start_frame)
        end_frame = self._clamp_frame(end_frame)
        if start_frame >= end_frame:
            self._warn("Episode playback requires start_frame < end_frame.")
            return

        self.playback_stop_frame = end_frame
        self._show_frame(start_frame)
        self._reset_playback_clock()
        self.play_timer.start()
        self._update_play_button()

    def _reset_playback_clock(self) -> None:
        self.playback_start_frame = self.current_frame
        self.play_clock.restart()

    def _set_playback_rate(self, *_args) -> None:  # noqa: ANN002
        if not hasattr(self, "speed_combo"):
            return
        rate = self.speed_combo.currentData()
        self.playback_rate = float(rate) if rate else 1.0
        self._update_play_timer_interval()
        if self.play_timer.isActive():
            self._reset_playback_clock()
        self.statusBar().showMessage(f"Playback speed: {self.playback_rate:.1f}x")

    def _update_play_timer_interval(self) -> None:
        if self.fps <= 0:
            self.play_timer.setInterval(15)
            return
        effective_fps = max(1.0, self.fps * self.playback_rate)
        frame_interval_ms = max(1, int(round(1000 / effective_fps)))
        self.play_timer.setInterval(max(1, min(15, frame_interval_ms // 2 or 1)))

    def _on_slider_value_changed(self, value: int) -> None:
        if self._updating_slider or self.capture is None:
            return
        self._seek_for_user(value)

    def _refresh_info(self) -> None:
        if self.video_path is None:
            self.filename_value.setText("-")
            self.path_value.setText("-")
            self.fps_value.setText("-")
            self.total_value.setText("-")
            self.current_frame_value.setText("-")
            self.current_time_value.setText("-")
            return

        self.filename_value.setText(Path(self.video_path).name)
        self.path_value.setText(self.video_path)
        self.fps_value.setText(f"{self.fps:.3f}")
        self.total_value.setText(str(self.total_frames))
        self.current_frame_value.setText(f"{self.current_frame} / {max(0, self.total_frames - 1)}")
        current_time = self.current_frame / self.fps if self.fps > 0 else 0.0
        self.current_time_value.setText(f"{current_time:.3f} sec")

    def _refresh_mark_labels(self, *_args) -> None:  # noqa: ANN002
        if not hasattr(self, "start_frame_value"):
            return
        self.start_frame_value.setText(self._mark_text("start"))
        self.peak_frame_value.setText(self._mark_text("peak"))
        self.end_frame_value.setText(self._mark_text("end"))

    def _mark_text(self, name: str) -> str:
        value = self.current_marks.get(name)
        if (
            name == "peak"
            and value is None
            and hasattr(self, "label_combo")
            and not label_requires_peak(self.label_combo.currentText())
        ):
            return "- (optional)"
        return "-" if value is None else str(value)

    def _refresh_episode_table(self) -> None:
        if not hasattr(self, "episode_table"):
            return
        rows = self.store.episodes_for_video(self.video_path) if self.video_path else []
        self.current_episode_rows = rows
        self.episode_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.get("episode_id", ""),
                row.get("start_frame", ""),
                row.get("peak_frame", ""),
                row.get("end_frame", ""),
                row.get("main_label", ""),
                row.get("confidence", ""),
                row.get("usable_for_training", ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.episode_table.setItem(row_index, column, item)

    def _selected_episode_row(self) -> int | None:
        row_index = self.episode_table.currentRow()
        if 0 <= row_index < len(self.current_episode_rows):
            return row_index
        selected_ranges = self.episode_table.selectedRanges()
        if not selected_ranges:
            return None
        row_index = selected_ranges[0].topRow()
        if 0 <= row_index < len(self.current_episode_rows):
            return row_index
        return None

    def _select_episode_id(self, episode_id: str) -> None:
        for row_index, row in enumerate(self.current_episode_rows):
            if row.get("episode_id") == episode_id:
                self.episode_table.setCurrentCell(row_index, 0)
                self.episode_table.selectRow(row_index)
                return

    def _jump_to_table_episode(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self.current_episode_rows):
            return
        episode = self.current_episode_rows[row]
        self._load_episode_into_form(episode)
        frame_index = self.current_marks.get("start")
        if frame_index is None:
            return
        self._seek_for_user(frame_index)
        self.statusBar().showMessage(
            f"Loaded {episode.get('episode_id', 'episode')} and jumped to start frame {frame_index}"
        )

    def _load_episode_into_form(self, row: dict[str, str]) -> None:
        self.loaded_episode_id = row.get("episode_id") or None
        self.current_marks = {
            "start": _optional_int(row.get("start_frame", "")),
            "peak": _optional_int(row.get("peak_frame", "")),
            "end": _optional_int(row.get("end_frame", "")),
        }

        for name, frame_index in self.current_marks.items():
            self.slider.set_marker(name, frame_index)

        self.person_id_edit.setText(row.get("person_id", ""))
        self._set_combo_value(self.label_combo, row.get("main_label", ""))
        self._set_spinbox_value(self.confidence_spin, row.get("confidence", ""))
        self._set_spinbox_value(self.intensity_spin, row.get("intensity", ""))
        self._set_spinbox_value(self.eye_spin, row.get("eye_involvement", ""))
        self._set_spinbox_value(self.mouth_spin, row.get("mouth_movement", ""))
        self._set_spinbox_value(self.cheek_spin, row.get("cheek_raise", ""))
        self._set_combo_value(self.symmetry_combo, row.get("symmetry", ""))
        self._set_combo_value(self.visible_quality_combo, row.get("visible_quality", ""))
        self._set_combo_value(self.usable_combo, row.get("usable_for_training", ""))
        self.note_edit.setPlainText(row.get("note", ""))
        self._refresh_mark_labels()

    def _set_combo_value(self, combo: QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_spinbox_value(self, spinbox: QSpinBox, value: str) -> None:
        try:
            spinbox.setValue(int(value))
        except (TypeError, ValueError):
            return

    def _prepare_next_episode_start(self, frame_index: int) -> None:
        self.loaded_episode_id = None
        self.current_marks = {"start": frame_index, "peak": None, "end": None}
        self.slider.clear_markers()
        self.slider.set_marker("start", frame_index)
        self._refresh_mark_labels()
        self.note_edit.clear()

    def _apply_default_usable(self, *_args) -> None:  # noqa: ANN002
        if not hasattr(self, "usable_combo"):
            return
        value = default_usable_for_training(
            self.confidence_spin.value(),
            self.visible_quality_combo.currentText(),
            self.label_combo.currentText(),
        )
        self.usable_combo.setCurrentText(value)

    def _update_play_button(self) -> None:
        if not hasattr(self, "play_button"):
            return
        if self.play_timer.isActive():
            self.play_button.setText("Pause")
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.play_button.setText("Play")
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def _clamp_frame(self, frame_index: int) -> int:
        if self.total_frames <= 0:
            return 0
        return max(0, min(int(frame_index), self.total_frames - 1))

    def _warn(self, message: str) -> None:
        QMessageBox.warning(self, "Annotation Tool", message)
        self.statusBar().showMessage(message)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()


def main() -> int:
    app = QApplication(sys.argv)
    window = SmileEpisodeAnnotationWindow()
    window.show()
    return app.exec()


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
