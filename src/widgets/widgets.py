from distutils import command
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QLabel, QPushButton, QLineEdit, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QValidator, QIntValidator, QDoubleValidator


class Colors:

    deep_orange = "#D35400"
    yellow = "#FFCC00"
    red = "#C0392B"
    orange = "#FF9C00"
    deep_blue = "#154360"
    blue = "#0099FF"
    dark_gray = "#1C2833"
    white = "#EAECEE"
    green = "#0BB900"
    __slots__ = ()


class TextStyle:

    title_1 = f"""
        color:{Colors.yellow};
        font-size:20px;
        font-weight:bold;
        """

    title_2 = f"""
        color:{Colors.yellow};
        font-size:14px;
        font-weight:600;
        """

    paragraph_1 = f"""
                    color:{Colors.white};
                    font-size:14px;
                    font-weight:700;
                """
    paragraph_2 = f"""
                    color:{Colors.white};
                    font-size:14px;
                    font-weight:500;
                """
    paragraph_3 = f"""
                    color:{Colors.white};
                    font-size:10px;
                    font-weight:500;
                """
    yellow_highlight = f"""
        color:{Colors.dark_gray};
        background-color:{Colors.yellow};
        font-size:22px;
        font-weight:bold;
        text-align:center;
        padding:20px 20px;
    """

    green_highlight = f"""
        color:{Colors.dark_gray};
        background-color:{Colors.green};
        font-size:20px;
        font-weight:bold;
        text-align:center;
        padding:20px 15px;
    """


class Text(QLabel):

    def __init__(self, label, style, *args, **kwargs):
        super().__init__(label, *args, **kwargs)
        self.setStyleSheet(style)


class Button(QPushButton):

    def __init__(self, label, command, *args, **kwargs):
        super().__init__(label, *args, **kwargs)
        self.clicked.connect(command)
        self.setStyleSheet(f"""
            color:{Colors.dark_gray};
            background-color:{Colors.white};
            padding:7px 0px;
            font-weight:bold;
            """)


class Input(QLineEdit):
    def __init__(self, placeholder=None, validator="int", text=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignLeft)
        self.setPlaceholderText(placeholder)

        if validator == "unlimit int":
            self.setValidator(QIntValidator())
        elif validator == "int":
            self.setValidator(QIntValidator(1, 9, self))
        elif validator == "float":
            self.setValidator(QDoubleValidator())

        self.setStyleSheet(f"""
                            color:{Colors.white};
                            border:1px solid  {Colors.orange};
                            border-radius:2;
                            font-size:14px;
                            height:40px;
                            margin-bottom:10px;
                            padding:2px 3px;
                            """
                           )

    def get_text(self) -> str:

        return self.text()

    def set_text(self, text):
        self.setText(str(text))
