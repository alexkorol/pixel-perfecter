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
from typing import Dict, List, Optional, Tuple

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
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QComboBox,
)

try:
    from ml.inference import MLSuggestion, suggest_parameters  # type: ignore
    ML_AVAILABLE = True
    ML_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pylint: disable=broad-except
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = exc

    @dataclass
    class MLSuggestion:  # type: ignore[override]
        """Fallback suggestion container when ML module is unavailable."""

        cell_size: int
        offset: Tuple[int, int]
        confidence: float = 0.0

    def suggest_parameters(path: Path) -> List[MLSuggestion]:  # type: ignore[override]
        raise RuntimeError("ML suggestions unavailable") from exc
from .reconstructor import PixelArtReconstructor, build_validation_diagnostics


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
    overlays: Dict[str, np.ndarray]
    cell_size: Optional[int]
    offset: Optional[Tuple[int, int]]
    metrics: dict
    warnings: List[str]
    mode: str
    region_summaries: List[dict] = field(default_factory=list)
    suggestions: List[MLSuggestion] = field(default_factory=list)


class ReconstructionWorker(QThread):
    result_ready = Signal(object)
    error = Signal(str)
    progress = Signal(int, int, str)

    def __init__(
        self,
        image_path: Path,
        override_cell_size: Optional[int],
        override_offset: Optional[Tuple[int, int]],
        apply_refinement: bool,
        use_ml: bool,
        adaptive_mode: bool,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.override_cell_size = override_cell_size
        self.override_offset = override_offset
        self.apply_refinement = apply_refinement
        self.use_ml = use_ml
        self.adaptive_mode = adaptive_mode
        self.debug = debug

    def run(self) -> None:
        try:
            reconstructor = PixelArtReconstructor(str(self.image_path), debug=self.debug)
            reconstructor.progress_callback = (
                lambda current, total, label: self.progress.emit(current, total, label)
            )
            mode = "adaptive" if self.adaptive_mode else "global"
            reconstruction = reconstructor.run(mode=mode)
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

            diagnostics = build_validation_diagnostics(
                str(self.image_path),
                reconstruction,
                cell_size,
                offset,
            )
            overlays: Dict[str, np.ndarray] = diagnostics["overlays"]  # type: ignore[assignment]
            metrics = diagnostics["metrics"]  # type: ignore[assignment]
            warnings = list(metrics.get("warnings", []))

            suggestions: List[MLSuggestion] = []
            if self.use_ml and ML_AVAILABLE:
                try:
                    suggestions = suggest_parameters(self.image_path)
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"ML suggestion error for {self.image_path}: {exc}")
            elif self.use_ml and not ML_AVAILABLE:
                print(f"ML suggestions requested but unavailable: {ML_IMPORT_ERROR}")

            payload = ReconstructionPayload(
                original=reconstructor.image,
                reconstruction=reconstruction,
                overlays=overlays,
                cell_size=int(cell_size) if cell_size else None,
                offset=tuple(offset) if offset else None,
                metrics=metrics,
                warnings=warnings,
                mode=mode,
                region_summaries=reconstructor.region_summaries,
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
        self.setWindowTitle("Pixel Perfecter - Interactive Preview")
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
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)

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
        open_btn = QPushButton("Open Image...")
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

        self.adaptive_check = QCheckBox("Adaptive per-region grid")
        control_layout.addWidget(self.adaptive_check)

        self.use_ml_check = QCheckBox("Fetch ML suggestions (if available)")
        self.use_ml_check.setChecked(True)
        if not ML_AVAILABLE:
            self.use_ml_check.setChecked(False)
            self.use_ml_check.setEnabled(False)
            tooltip = "ML suggestions unavailable."
            if ML_IMPORT_ERROR:
                tooltip += f" ({ML_IMPORT_ERROR})"
            self.use_ml_check.setToolTip(tooltip)
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
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Diagnostic view:"))
        self.overlay_mode_combo = QComboBox()
        self.overlay_mode_combo.addItem("Combined (core+halo)", "combined")
        self.overlay_mode_combo.addItem("Raw (all differences)", "raw")
        self.overlay_mode_combo.addItem("Halo-suppressed", "halo_suppressed")
        self.overlay_mode_combo.addItem("Core-only", "core_only")
        self.overlay_mode_combo.currentIndexChanged.connect(self._update_overlay_mode)
        mode_row.addWidget(self.overlay_mode_combo)
        metrics_layout.addLayout(mode_row)
        vbox.addWidget(metrics_box)

        # Feedback
        feedback_box = QGroupBox("Feedback")
        feedback_layout = QVBoxLayout(feedback_box)
        self.feedback_combo = QComboBox()
        self.feedback_combo.addItems(["Unreviewed", "Looks correct", "Incorrect"])
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes...")
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
            self.overlay_mode_combo.blockSignals(True)
            self.overlay_mode_combo.setCurrentIndex(0)
            self.overlay_mode_combo.blockSignals(False)
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
            self.adaptive_check,
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
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Preparing refinement...")
        self.progress_bar.show()
        self.populate_suggestions([])

        self.worker = ReconstructionWorker(
            image_path=self.image_path,
            override_cell_size=override_cell,
            override_offset=override_offset,
            apply_refinement=self.apply_refine_check.isChecked(),
            use_ml=self.use_ml_check.isChecked(),
            adaptive_mode=self.adaptive_check.isChecked(),
        )
        self.worker.result_ready.connect(self.handle_result)
        self.worker.error.connect(self.handle_error)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def update_progress(self, current: int, total: int, label: str) -> None:
        if total <= 0:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat(label or "Refining grid...")
            self.status_label.setText("Refining grid...")
        else:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(current, total))
            self.progress_bar.setFormat(f"{label} ({current}/{total})")
            self.status_label.setText(f"Refining grid... {current}/{total}")
        if not self.progress_bar.isVisible():
            self.progress_bar.show()

    def _on_worker_finished(self) -> None:
        self.set_controls_enabled(True)
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")

    def _update_overlay_mode(self) -> None:
        if not self.current_payload:
            return
        mode_data = self.overlay_mode_combo.currentData()
        mode = mode_data if isinstance(mode_data, str) else "combined"
        overlays = self.current_payload.overlays
        overlay = overlays.get(mode)
        if overlay is None:
            overlay = overlays.get("combined")
        if overlay is not None:
            self.overlay_label.set_image(overlay)

    def handle_result(self, payload: ReconstructionPayload) -> None:
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.current_payload = payload
        self.original_label.set_image(payload.original)
        # Reset combo to default and render overlay matching selection
        self.overlay_mode_combo.blockSignals(True)
        self.overlay_mode_combo.setCurrentIndex(0)
        self.overlay_mode_combo.blockSignals(False)
        self._update_overlay_mode()
        upscaled = cv2.resize(
            payload.reconstruction,
            (payload.original.shape[1], payload.original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        self.recon_label.set_image(upscaled)
        metrics = payload.metrics
        mode_label = "Adaptive per-region" if payload.mode == "adaptive" else "Global grid"
        cell_label = f"{payload.cell_size}px" if payload.cell_size is not None else "varies"
        offset_label = (
            f"({payload.offset[0]}, {payload.offset[1]})"
            if payload.offset is not None
            else "n/a"
        )
        metrics_text = (
            f"Mode: {mode_label}\n"
            f"Cell size: {cell_label}\n"
            f"Offset: {offset_label}\n"
            f"Diff total: {metrics.get('percent_diff_total', metrics.get('percent_diff', 0.0)):.2f}%\n"
            f"Diff core: {metrics.get('percent_diff_core', 0.0):.2f}%\n"
            f"Diff halo: {metrics.get('percent_diff_halo', 0.0):.2f}%\n"
            f"Diff outside: {metrics.get('percent_diff_outside', 0.0):.2f}%\n"
            f"Grid ratio: {metrics.get('grid_ratio', 0.0):.3f}\n"
            f"Warnings: {', '.join(payload.warnings) if payload.warnings else 'None'}"
        )
        if payload.mode == "adaptive" and payload.region_summaries:
            lines: List[str] = []
            for summary in payload.region_summaries[:6]:
                status = summary.get("status", "ok")
                size = summary.get("cell_size")
                offset = summary.get("offset")
                core = summary.get("core_diff")
                size_txt = f"{size}px" if size else "n/a"
                offset_txt = f"{offset}" if offset else "n/a"
                core_txt = f"{core:.2f}%" if core is not None else "n/a"
                lines.append(
                    f"#{summary.get('index', '?')} ({status}) size {size_txt}, offset {offset_txt}, core {core_txt}"
                )
            metrics_text += "\nRegions:\n" + "\n".join(lines)
        self.metrics_label.setText(metrics_text)
        self.status_label.setText("Reconstruction complete.")
        self.populate_suggestions(payload.suggestions)

    def handle_error(self, message: str) -> None:
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
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
            "percent_diff_total",
            "percent_diff_core",
            "percent_diff_halo",
            "percent_diff_outside",
            "percent_diff",
            "grid_ratio",
            "warnings",
            "notes",
            "override_cell_size",
            "override_offset_x",
            "override_offset_y",
            "ml_suggestions",
            "ml_selected_index",
            "mode",
            "region_summaries",
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
            metrics = self.current_payload.metrics
            total_diff = metrics.get("percent_diff_total", metrics.get("percent_diff", 0.0))
            warnings_str = " | ".join(self.current_payload.warnings)
            writer.writerow(
                {
                    "timestamp": dt.datetime.utcnow().isoformat(),
                    "image": str(self.image_path),
                    "feedback": feedback,
                    "cell_size": self.current_payload.cell_size
                    if self.current_payload.cell_size is not None
                    else "",
                    "offset_x": self.current_payload.offset[0]
                    if self.current_payload.offset is not None
                    else "",
                    "offset_y": self.current_payload.offset[1]
                    if self.current_payload.offset is not None
                    else "",
                    "percent_diff_total": round(total_diff, 4),
                    "percent_diff_core": round(metrics.get("percent_diff_core", 0.0), 4),
                    "percent_diff_halo": round(metrics.get("percent_diff_halo", 0.0), 4),
                    "percent_diff_outside": round(metrics.get("percent_diff_outside", 0.0), 4),
                    "percent_diff": round(total_diff, 4),
                    "grid_ratio": round(metrics.get("grid_ratio", 0.0), 4),
                    "warnings": warnings_str,
                    "notes": notes,
                    "override_cell_size": override_cell,
                    "override_offset_x": override_offset_x,
                    "override_offset_y": override_offset_y,
                    "ml_suggestions": suggestions_str,
                    "ml_selected_index": selected_idx,
                    "mode": self.current_payload.mode,
                    "region_summaries": repr(self.current_payload.region_summaries),
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

