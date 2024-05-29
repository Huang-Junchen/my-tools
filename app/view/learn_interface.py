from PyQt5.QtWidgets import QWidget, QGridLayout
from PyQt5.QtCore import pyqtSlot

from contextlib import redirect_stdout
with redirect_stdout(None):
    from qfluentwidgets import ScrollArea, PushButton, FluentIcon

from app.common.style_sheet import StyleSheet


class LearnInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QGridLayout(self)

        self.btn = PushButton(FluentIcon.FOLDER, 'Click Me')
        self.btn.clicked.connect(self.on_click)

        self.view.addWidget(self.btn, 0, 0)

        self.__initWidget()
        # self.loadSamples()

    def __initWidget(self):
        self.view.setObjectName('view')
        self.setObjectName('learnInterface')
        StyleSheet.HOME_INTERFACE.apply(self)

    @pyqtSlot()
    def on_click(self):
        print('Button clicked!')
