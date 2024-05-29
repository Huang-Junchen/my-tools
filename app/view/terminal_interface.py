import locale
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QPlainTextEdit
from PyQt5.QtCore import QProcess, QProcessEnvironment
from qfluentwidgets import ScrollArea
from PyQt5.QtCore import pyqtSlot

from app.common.style_sheet import StyleSheet


class TerminalInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__initWidget()
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)

    def __initWidget(self):
        self.setObjectName('terminalInterface')
        StyleSheet.HOME_INTERFACE.apply(self)

        self.centralWidget = QWidget(self)
        self.setWidget(self.centralWidget)
        self.setWidgetResizable(True)

        self.layout = QVBoxLayout(self.centralWidget)

        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

        self.runButton = QPushButton("Run Command", self)
        self.runButton.clicked.connect(self.run_command)
        self.layout.addWidget(self.runButton)

        self.layout.setStretch(0, 1)
        self.layout.setStretch(1, 0)

    def run_command(self):
        self.output.clear()
        self.process.setProcessEnvironment(QProcessEnvironment.systemEnvironment())
        self.process.start("cmd", ["/c", "ping -n 4 www.baidu.com"])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data()
        text = data.decode(locale.getpreferredencoding(), errors='ignore')
        self.output.appendPlainText(text)

    def handle_stderr(self):
        data = self.process.readAllStandardError().data()
        text = data.decode(locale.getpreferredencoding(), errors='ignore')
        self.output.appendPlainText(text)

    def process_finished(self):
        self.output.appendPlainText("Process finished")

    def process_error(self, error):
        self.output.appendPlainText(f"Process error: {error}")