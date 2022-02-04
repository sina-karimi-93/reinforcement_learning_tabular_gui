import sys
from src import QApplication, QMainWindow, MainFrame, Colors
from src import clear_terminal, install_dependencies


class MainWindow(QMainWindow):
    """
    MainWindow which contains MainFrame.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Reinforcement Learning - Q-Learning")

        welcome_frame = MainFrame()
        self.setCentralWidget(welcome_frame)
        self.setStyleSheet(f"background-color:{Colors.dark_gray};")


if __name__ == "__main__":
    clear_terminal()
    install_dependencies()
    clear_terminal()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
