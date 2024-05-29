from PyQt5.QtCore import Qt, QSize, QEvent
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from contextlib import redirect_stdout

from .terminal_interface import TerminalInterface

with redirect_stdout(None):
    from qfluentwidgets import NavigationItemPosition, setThemeColor, setTheme, Theme, \
        SplashScreen, NavigationBarPushButton, toggleTheme, MSFluentWindow
    from qfluentwidgets import FluentIcon as FIF

from .home_interface import HomeInterface
from .learn_interface import LearnInterface

class MainWindow(MSFluentWindow):
    """ 主界面 """

    def __init__(self):
        super().__init__()

        self.initWindow()
        self.initInterface()
        self.initNavigation()

    def initWindow(self):
        setThemeColor('#f18cb9', lazy=True)
        setTheme(Theme.AUTO, lazy=True)
        self.setMicaEffectEnabled(False)

        self.resize(960, 640)
        self.setWindowIcon(QIcon('./app/assets/images/icon.ico'))
        self.setWindowTitle("My Tools")

        self.disableMaximize()

        # 创建启动画面
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(128, 128))
        self.splashScreen.raise_()

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

        self.show()
        QApplication.processEvents()

    def initInterface(self):
        self.homeInterface = HomeInterface(self)
        self.learnInterface = LearnInterface(self)
        self.terminalInterface = TerminalInterface(self)

    def initNavigation(self):
        self.addSubInterface(self.homeInterface, FIF.HOME, self.tr('主页'))
        self.addSubInterface(self.learnInterface, FIF.EDUCATION, self.tr('学习'))

        self.navigationInterface.addWidget(
            'startButton',
            NavigationBarPushButton(FIF.PLAY, '启动', isSelectable=False),
            lambda: toggleTheme(lazy=True),
            NavigationItemPosition.BOTTOM)

        self.navigationInterface.addWidget(
            'themeButton',
            NavigationBarPushButton(FIF.BRUSH, '主题', isSelectable=False),
            lambda: toggleTheme(lazy=True),
            NavigationItemPosition.BOTTOM)

        self.addSubInterface(self.terminalInterface, FIF.COMMAND_PROMPT, self.tr('终端'), position=NavigationItemPosition.BOTTOM)
        # self.addSubInterface(self.settingInterface, FIF.SETTING, self.tr('设置'), position=NavigationItemPosition.BOTTOM)

        self.splashScreen.finish()

    def disableMaximize(self):
        # 禁用最大化
        self.titleBar.maxBtn.setHidden(True)
        self.titleBar.maxBtn.setDisabled(True)
        self.titleBar.setDoubleClickEnabled(False)
        self.setResizeEnabled(False)
        # self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
