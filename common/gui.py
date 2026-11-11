from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)


class _Worker(QObject):
    log = Signal(str)
    succeeded = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, runner: Callable, config_path: str):
        super().__init__()
        self.runner = runner
        self.config_path = config_path

    @Slot()
    def run(self):
        try:
            result = self.runner(self.config_path, self.log.emit)
            self.succeeded.emit(f"任务完成\n{result}")
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()


class PipelineWindow:
    """Shared PySide6 shell; each stage supplies its own pipeline runner."""

    def __init__(self, title: str, default_config: Path, runner: Callable, note: str):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.runner = runner
        self.thread: QThread | None = None
        self.worker: _Worker | None = None
        self.child_windows: list[PipelineWindow] = []
        self.window = QMainWindow()
        self.window.setWindowTitle(title)
        self.window.resize(960, 680)
        central = QWidget()
        self.window.setCentralWidget(central)
        layout = QVBoxLayout(central)

        description = QLabel(note)
        description.setWordWrap(True)
        layout.addWidget(description)
        row = QHBoxLayout()
        row.addWidget(QLabel("配置文件"))
        self.config_edit = QLineEdit(str(default_config))
        row.addWidget(self.config_edit, 1)
        browse = QPushButton("浏览")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self._start)
        row.addWidget(self.start_button)
        self.action_row = row
        layout.addLayout(row)

        self.status = QLabel("就绪")
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("font-family: Consolas, 'Microsoft YaHei'; font-size: 10pt;")
        layout.addWidget(self.log_view, 1)

    def add_pipeline_action(
        self,
        label: str,
        title: str,
        default_config: Path,
        runner: Callable,
        note: str,
    ) -> None:
        button = QPushButton(label)

        def open_child():
            child = PipelineWindow(title, default_config, runner, note)
            child.window.destroyed.connect(lambda *_: self._forget_child(child))
            self.child_windows.append(child)
            child.window.show()

        button.clicked.connect(open_child)
        self.action_row.insertWidget(max(self.action_row.count() - 1, 0), button)

    def _forget_child(self, child: "PipelineWindow"):
        if child in self.child_windows:
            self.child_windows.remove(child)

    @Slot()
    def _browse(self):
        selected, _ = QFileDialog.getOpenFileName(
            self.window, "选择配置", self.config_edit.text(), "YAML (*.yaml *.yml);;All files (*)"
        )
        if selected:
            self.config_edit.setText(selected)

    @Slot()
    def _start(self):
        config_path = Path(self.config_edit.text())
        if not config_path.exists():
            QMessageBox.critical(self.window, "配置错误", f"配置文件不存在：{config_path}")
            return
        self.start_button.setEnabled(False)
        self.status.setText("运行中")
        self.progress.setRange(0, 0)
        self.log_view.appendPlainText(f"开始：{config_path}")
        self.thread = QThread(self.window)
        self.worker = _Worker(self.runner, str(config_path))
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.log_view.appendPlainText)
        self.worker.succeeded.connect(self._success)
        self.worker.failed.connect(self._failure)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    @Slot(str)
    def _success(self, message: str):
        self.log_view.appendPlainText(message)
        self.status.setText("完成")

    @Slot(str)
    def _failure(self, message: str):
        self.log_view.appendPlainText(message)
        self.status.setText("失败")
        QMessageBox.critical(self.window, "任务失败", "详细错误已写入窗口日志")

    @Slot()
    def _thread_finished(self):
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if self.status.text() == "完成" else 0)
        self.start_button.setEnabled(True)
        self.worker = None
        self.thread = None

    def mainloop(self) -> int:
        self.window.show()
        return self.app.exec()
