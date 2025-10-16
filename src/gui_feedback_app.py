"""
Interactive GUI for Pixel Perfecter experimentation.

Provides drag & drop preview, live reconstruction controls, ML suggestions,
and structured feedback logging to accelerate manual inspection.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, QSize, Qt, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QComboBox,
)

from ml.inference import MLSuggestion, suggest_parameters
from .pixel_reconstructor import PixelArtReconstructor, create_validation_overlay


def numpy_to_qimage(array: np.ndarray) -> QImage:
    """Convert an RGB numpy array into a QImage."""
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected RGB array for display.")
    height, width, _ = array.shape
    bytes_per_line = 3 * width
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return QImage(
        array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
    ).copy()


@dataclass
class ReconstructionPayload:
    original: np.ndarray
    reconstruction: np.ndarray
    overlay: np.ndarray
    cell_size: int
    offset: Tuple[int, int]
    percent_diff: float
    grid_ratio: float
    warnings: str
    suggestions: List[MLSuggestion] = field(default_factory=list)


class ReconstructionWorker(QThread):
    result_ready = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        image_path: Path,
        override_cell_size: Optional[int],
        override_offset: Optional[Tuple[int, int]],
        apply_refinement: bool,
        use_ml: bool,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.override_cell_size = override_cell_size
        self.override_offset = override_offset
        self.apply_refinement = apply_refinement
        self.use_ml = use_ml
        self.debug = debug

    def run(self) -> None:
        try:
            reconstructor = PixelArtReconstructor(str(self.image_path), debug=self.debug)
            reconstruction = reconstructor.run()
            cell_size = reconstructor.cell_size
            offset = reconstructor.offset

            if self.override_cell_size is not None or self.override_offset is not None:
                if self.override_cell_size is not None:
                    cell_size = self.override_cell_size
                if self.override_offset is not None:
                    offset = self.override_offset
                reconstructor.cell_size = cell_size
                reconstructor.offset = offset
                reconstruction = reconstructor._empirical_pixel_reconstruction()

            if self.apply_refinement:
                reconstructor.grid_result = reconstruction
                reconstruction = reconstructor._refine_artifacts()

            overlay = create_validation_overlay(
                str(self.image_path),
                reconstruction,
                cell_size,
                offset,
            )

            overlay_np = np.array(overlay)
            red_mask = (
                (overlay_np[..., 0] == 255)
                & (overlay_np[..., 1] == 0)
                & (overlay_np[..., 2] == 0)
            )
            percent_diff = 100.0 * np.sum(red_mask) / (
                overlay_np.shape[0] * overlay_np.shape[1]
            )
            grid_ratio = cell_size / min(overlay_np.shape[0], overlay_np.shape[1])
            warnings = []
            if percent_diff > 10.0:
                warnings.append(f"High difference: {percent_diff:.1f}% pixels differ.")
            if grid_ratio > 0.25:
                warnings.append(
                    f"Grid size {cell_size} large relative to image ({grid_ratio:.2f})."
                )
            if cell_size < 4:
                warnings.append("Grid size appears very small (possible noise).")

            suggestions: List[MLSuggestion] = []
            if self.use_ml:
                try:
                    suggestions = suggest_parameters(self.image_path)
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"ML suggestion error for {self.image_path}: {exc}")

            payload = ReconstructionPayload(
                original=reconstructor.image,
                reconstruction=reconstruction,
                overlay=overlay,
                cell_size=int(cell_size),
                offset=offset,
                percent_diff=float(percent_diff),
                grid_ratio=float(grid_ratio),
                warnings=" | ".join(warnings),
                suggestions=suggestions,
            )
            self.result_ready.emit(payload)
        except Exception as exc:  # pylint: disable=broad-except
            self.error.emit(str(exc))


class PreviewLabel(QLabel):
    """QLabel specialised for displaying scaled pixmaps."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.title = title
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(320, 320))
        self.setText(f"\n\n{title}\n\nDrop an image or use 'Open Image'")

    def set_image(self, array: np.ndarray) -> None:
        pixmap = QPixmap.fromImage(numpy_to_qimage(array))
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        pixmap = self.pixmap()
        if pixmap:
            self.setPixmap(
                pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pixel Perfecter — Interactive Preview")
        self.resize(1400, 820)

        self.image_path: Optional[Path] = None
        self.current_payload: Optional[ReconstructionPayload] = None
        self.worker: Optional[ReconstructionWorker] = None

        self.current_suggestions: List[MLSuggestion] = []
        self.selected_suggestion_idx: Optional[int] = None
        self._suppress_override_signal = False

        self.original_label = PreviewLabel("Original")
        self.overlay_label = PreviewLabel("Overlay")
        self.recon_label = PreviewLabel("Reconstructed")

        self.status_label = QLabel("Ready.")
        self.statusBar().addWidget(self.status_label)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout()
        central.setLayout(layout)

        image_grid = QGridLayout()
        image_grid.addWidget(self.original_label, 0, 0)
        image_grid.addWidget(self.overlay_label, 0, 1)
        image_grid.addWidget(self.recon_label, 1, 0, 1, 2)

        layout.addLayout(image_grid, stretch=2)
        layout.addWidget(self._build_controls_panel(), stretch=1)

        self.setAcceptDrops(True)
        self._create_menu()

    # ----------------------- UI construction helpers -----------------------

    def _create_menu(self) -> None:
        open_action = QAction("&Open Image", self)
        open_action.triggered.connect(self.open_image_dialog)
        self.menuBar().addAction(open_action)

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        vbox = QVBoxLayout(panel)

        # Image selection
        path_box = QGroupBox("Image")
        path_layout = QVBoxLayout(path_box)
        self.path_field = QLineEdit()
        self.path_field.setReadOnly(True)
        open_btn = QPushButton("Open Image…")
        open_btn.clicked.connect(self.open_image_dialog)
        path_layout.addWidget(self.path_field)
        path_layout.addWidget(open_btn)
        vbox.addWidget(path_box)

        # Override controls
        control_box = QGroupBox("Overrides")
        control_layout = QVBoxLayout(control_box)

        cell_row = QHBoxLayout()
        self.cell_size_spin = QSpinBox()
        self.cell_size_spin.setRange(0, 256)
        cell_row.addWidget(QLabel("Cell size (0 = auto):"))
        cell_row.addWidget(self.cell_size_spin)
        control_layout.addLayout(cell_row)

        quick_row = QHBoxLayout()
        half_btn = QPushButton("Half")
        half_btn.clicked.connect(lambda: self._adjust_cell_size(0.5))
        double_btn = QPushButton("Double")
        double_btn.clicked.connect(lambda: self._adjust_cell_size(2.0))
        quick_row.addWidget(half_btn)
        quick_row.addWidget(double_btn)
        control_layout.addLayout(quick_row)

        offset_row = QHBoxLayout()
        self.offset_x_spin = QSpinBox()
        self.offset_x_spin.setRange(0, 256)
        self.offset_y_spin = QSpinBox()
        self.offset_y_spin.setRange(0, 256)
        offset_row.addWidget(QLabel("Offset X:"))
        offset_row.addWidget(self.offset_x_spin)
        offset_row.addWidget(QLabel("Offset Y:"))
        offset_row.addWidget(self.offset_y_spin)
        control_layout.addLayout(offset_row)

        self.apply_refine_check = QCheckBox("Apply artifact refinement")
        control_layout.addWidget(self.apply_refine_check)

        self.use_ml_check = QCheckBox("Fetch ML suggestions (if available)")
        self.use_ml_check.setChecked(True)
        control_layout.addWidget(self.use_ml_check)

        run_btn = QPushButton("Run Reconstruction")
        run_btn.clicked.connect(self.run_reconstruction)
        control_layout.addWidget(run_btn)

        vbox.addWidget(control_box)

        # Metrics
        metrics_box = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_box)
        self.metrics_label = QLabel("No data yet.")
        self.metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.metrics_label)
        vbox.addWidget(metrics_box)

        # Feedback
        feedback_box = QGroupBox("Feedback")
        feedback_layout = QVBoxLayout(feedback_box)
        self.feedback_combo = QComboBox()
        self.feedback_combo.addItems(["Unreviewed", "Looks correct", "Incorrect"])
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes…")
        save_btn = QPushButton("Save Feedback")
        save_btn.clicked.connect(self.save_feedback)
        feedback_layout.addWidget(self.feedback_combo)
        feedback_layout.addWidget(self.notes_edit)
        feedback_layout.addWidget(save_btn)
        vbox.addWidget(feedback_box)

        # Suggestions
        suggestions_box = QGroupBox("ML Suggestions")
        suggestions_layout = QVBoxLayout(suggestions_box)
        self.suggestions_list = QListWidget()
        self.apply_suggestion_btn = QPushButton("Apply Selected Suggestion")
        self.apply_suggestion_btn.setEnabled(False)
        self.apply_suggestion_btn.clicked.connect(self.apply_selected_suggestion)
        suggestions_layout.addWidget(self.suggestions_list)
        suggestions_layout.addWidget(self.apply_suggestion_btn)
        vbox.addWidget(suggestions_box)

        vbox.addStretch(1)

        self.cell_size_spin.valueChanged.connect(self._manual_override_changed)
        self.offset_x_spin.valueChanged.connect(self._manual_override_changed)
        self.offset_y_spin.valueChanged.connect(self._manual_override_changed)

        return panel

    # ----------------------------- Drag & drop -----------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file():
                self.load_image(path)
                break
        event.acceptProposedAction()

    # ----------------------------- Image loading ---------------------------

    def open_image_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input image",
            str(Path("input").resolve()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)",
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        self.image_path = path
        self.path_field.setText(str(path))
        self.selected_suggestion_idx = None
        try:
            with Image.open(path) as img:
                image = np.array(img.convert("RGB"))
            self.original_label.set_image(image)
            self.overlay_label.setText("\n\nRun reconstruction to view overlay")
            self.recon_label.setText("\n\nRun reconstruction to view result")
            self.metrics_label.setText("No reconstruction metrics yet.")
            self.status_label.setText("Image loaded. Ready.")
            self.current_payload = None
            self.populate_suggestions([])
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{exc}")

    # ------------------------------ Actions --------------------------------

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.cell_size_spin,
            self.offset_x_spin,
            self.offset_y_spin,
            self.apply_refine_check,
            self.use_ml_check,
            self.suggestions_list,
            self.apply_suggestion_btn,
        ):
            widget.setEnabled(enabled)

    def run_reconstruction(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "No image selected", "Please load an image first.")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self, "Busy", "Reconstruction already in progress. Please wait."
            )
            return

        override_cell = self.cell_size_spin.value() or None
        override_offset = (
            (self.offset_x_spin.value(), self.offset_y_spin.value())
            if any((self.offset_x_spin.value(), self.offset_y_spin.value()))
            else None
        )

        self.set_controls_enabled(False)
        self.status_label.setText("Running reconstruction...")
        self.populate_suggestions([])

        self.worker = ReconstructionWorker(
            image_path=self.image_path,
            override_cell_size=override_cell,
            override_offset=override_offset,
            apply_refinement=self.apply_refine_check.isChecked(),
            use_ml=self.use_ml_check.isChecked(),
        )
        self.worker.result_ready.connect(self.handle_result)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(lambda: self.set_controls_enabled(True))
        self.worker.start()

    def handle_result(self, payload: ReconstructionPayload) -> None:
        self.current_payload = payload
        self.original_label.set_image(payload.original)
        self.overlay_label.set_image(payload.overlay)
        upscaled = cv2.resize(
            payload.reconstruction,
            (payload.original.shape[1], payload.original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        self.recon_label.set_image(upscaled)
        metrics_text = (
            f"Cell size: {payload.cell_size}px\n"
            f"Offset: ({payload.offset[0]}, {payload.offset[1]})\n"
            f"Percent diff: {payload.percent_diff:.2f}%\n"
            f"Grid ratio: {payload.grid_ratio:.3f}\n"
            f"Warnings: {payload.warnings or 'None'}"
        )
        self.metrics_label.setText(metrics_text)
        self.status_label.setText("Reconstruction complete.")
        self.populate_suggestions(payload.suggestions)

    def handle_error(self, message: str) -> None:
        self.status_label.setText("Error during reconstruction.")
        QMessageBox.critical(self, "Reconstruction Error", message)

    # ------------------------------ Feedback --------------------------------

    def save_feedback(self) -> None:
        if not self.image_path or not self.current_payload:
            QMessageBox.warning(
                self, "No reconstruction", "Run a reconstruction before saving feedback."
            )
            return

        feedback = self.feedback_combo.currentText()
        notes = self.notes_edit.toPlainText().strip()
        override_cell = self.cell_size_spin.value() or ""
        override_offset_x = self.offset_x_spin.value() or ""
        override_offset_y = self.offset_y_spin.value() or ""

        suggestions_str = ";".join(
            f"{idx}:{suggestion.cell_size}:{suggestion.offset[0]}:"
            f"{suggestion.offset[1]}:{suggestion.confidence:.3f}"
            for idx, suggestion in enumerate(self.current_suggestions)
        )
        selected_idx = (
            str(self.selected_suggestion_idx)
            if self.selected_suggestion_idx is not None
            else ""
        )

        log_path = Path("notes/feedback_log.csv")
        fieldnames = [
            "timestamp",
            "image",
            "feedback",
            "cell_size",
            "offset_x",
            "offset_y",
            "percent_diff",
            "grid_ratio",
            "warnings",
            "notes",
            "override_cell_size",
            "override_offset_x",
            "override_offset_y",
            "ml_suggestions",
            "ml_selected_index",
        ]

        if log_path.exists():
            with log_path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                existing_header = next(reader, None)
            if existing_header != fieldnames:
                rows: List[dict] = []
                with log_path.open("r", newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    rows.extend(reader)
                with log_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({key: row.get(key, "") for key in fieldnames})

        is_new_file = not log_path.exists()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if is_new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": dt.datetime.utcnow().isoformat(),
                    "image": str(self.image_path),
                    "feedback": feedback,
                    "cell_size": self.current_payload.cell_size,
                    "offset_x": self.current_payload.offset[0],
                    "offset_y": self.current_payload.offset[1],
                    "percent_diff": round(self.current_payload.percent_diff, 4),
                    "grid_ratio": round(self.current_payload.grid_ratio, 4),
                    "warnings": self.current_payload.warnings,
                    "notes": notes,
                    "override_cell_size": override_cell,
                    "override_offset_x": override_offset_x,
                    "override_offset_y": override_offset_y,
                    "ml_suggestions": suggestions_str,
                    "ml_selected_index": selected_idx,
                }
            )

        QMessageBox.information(self, "Saved", "Feedback recorded.")
        self.status_label.setText("Feedback saved.")

    # ------------------------------ Suggestions -----------------------------

    def populate_suggestions(self, suggestions: List[MLSuggestion]) -> None:
        self.current_suggestions = suggestions
        self.selected_suggestion_idx = None
        self.suggestions_list.clear()

        if not suggestions:
            self.suggestions_list.addItem("No suggestions available.")
            self.apply_suggestion_btn.setEnabled(False)
            return

        for idx, suggestion in enumerate(suggestions, start=1):
            text = (
                f"#{idx}: cell {suggestion.cell_size}px | "
                f"offset ({suggestion.offset[0]}, {suggestion.offset[1]}) | "
                f"conf {suggestion.confidence:.2f}"
            )
            self.suggestions_list.addItem(text)

        self.suggestions_list.setCurrentRow(0)
        self.apply_suggestion_btn.setEnabled(True)

    def apply_selected_suggestion(self) -> None:
        row = self.suggestions_list.currentRow()
        if row < 0 or row >= len(self.current_suggestions):
            QMessageBox.information(self, "No selection", "Select a suggestion first.")
            return

        suggestion = self.current_suggestions[row]
        self._suppress_override_signal = True
        try:
            self.cell_size_spin.setValue(suggestion.cell_size)
            self.offset_x_spin.setValue(suggestion.offset[0])
            self.offset_y_spin.setValue(suggestion.offset[1])
        finally:
            self._suppress_override_signal = False

        self.selected_suggestion_idx = row
        self.status_label.setText(f"Applied ML suggestion #{row + 1}. Re-running.")
        self.run_reconstruction()

    def _adjust_cell_size(self, factor: float) -> None:
        current = self.cell_size_spin.value()
        if current == 0:
            return
        new_value = max(1, min(256, int(round(current * factor))))
        self.cell_size_spin.setValue(new_value)

    def _manual_override_changed(self) -> None:
        if self._suppress_override_signal:
            return
        self.selected_suggestion_idx = None

    # ----------------------------- Drag compat ------------------------------

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        event.accept()


def main() -> None:
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
