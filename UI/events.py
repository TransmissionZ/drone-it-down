import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QVBoxLayout, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer, QTime, Qt


def showTime(ui):
    # getting current time
    current_time = QTime.currentTime()

    # converting QTime object to string
    label_time = current_time.toString('hh:mm:ss')

    # showing it to the label
    ui.label.setText(label_time)


def sync_url(ui):
    url = 'rtmp://www.droneitdown.com/rtmp/' + ui
    return url
