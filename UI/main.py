import time

import cv2
import numpy
import requests

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QHBoxLayout
from PyQt5.QtCore import QDateTime, QRegExp
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QRegExpValidator, QMovie
import qdarktheme
import UI.resources
import geocoder

from UI.events import *
from PyQt5 import uic
import os
import numpy as np
from PyQt5.QtGui import QPixmap
from Backend.alignment import initialize_db
from Backend.inference import create_embedding_dataset
import yaml
import requests
from natsort import natsorted
import time

class NameThread(QThread):
    update_matched = pyqtSignal(list)

    def __init__(self, url):
        super().__init__()
        self._run_flag = True
        self.url = url
        self.start_time = time.time()
        self.base_url = "http://34.142.141.69/"

    def run(self):
        while self._run_flag:
            diff = int(time.time() - self.start_time)
            if diff % 2 == 0:
                url = self.base_url + 'get_matched/'
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        self.update_matched.emit(response.json())
                    else:
                        raise response.content
                except Exception as e:
                    print('Failed to fetch match count: ' + str(e))

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        # self.wait()
        del self


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_time = pyqtSignal(int)
    update_matched = pyqtSignal(list)

    def __init__(self, url):
        super().__init__()
        self._run_flag = True
        self.url = url
        self.start_time = time.time()
        self.FPS = 1 / 30
        self.FPS_MS = int(self.FPS * 1000)

    def run(self):
        # capture from webcam
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        c = 0
        while self._run_flag:
            ret, cv_img = cap.read()
            time.sleep(self.FPS)
            if ret:
                self.change_pixmap_signal.emit(cv_img)
                diff = int(time.time() - self.start_time)
                self.update_time.emit(diff)

            else:
                c += 1
                print(ret, cv_img)
                if not cap.isOpened():
                    self.change_pixmap_signal.emit(numpy.ndarray([0]))
                    self.update_time.emit(int(time.time() - self.start_time))
                    break
                if c > 1000:
                    self.change_pixmap_signal.emit(numpy.ndarray([0]))
                    break
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        # self.wait()
        del self


class MainWindow(QMainWindow):
    def __init__(self, CONFIG_FILE, URL, RTMP):
        super(MainWindow, self).__init__()
        self.current_list = []
        self.matched_dict = {}
        uic.loadUi("UI/main.ui", self)
        self.CONFIG_FILE = CONFIG_FILE
        self.config = self.get_app_config()
        # self.target_file = self.config['DATA']['TARGET']
        self.folder_pre = self.config['DATA']['PREPROCESSING_FOLDER']
        self.model_file = self.config['DATA']['MODEL_FILE']
        self.rtmp_url = RTMP
        self.base_url = URL
        self.modify()
        # self.start_video(2)
        self.setFixedSize(1200, 700)
        self.show()

        self.stream_url = self.base_url + 'video_feed/' + self.config['DATA']['USERNAME'] + '/' + self.config['DATA'][
            'MODEL_FILE']

    def get_app_config(self):
        with open(self.CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f)

        return config

    def set_app_config(self):
        with open(self.CONFIG_FILE, 'w') as file:
            yaml.dump(self.config, file)

    def selectPreFolder(self):
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.folder_pre = folderpath
        self.prefolder.setText('Selected Folder: ' + folderpath)

    def selectFile(self):
        file = QtWidgets.QFileDialog.getOpenFileName(self, 'Select a File')
        self.target_file = file
        self.target.setText('Selected File: ' + file[0])

    def updateCompanyImage(self):
        #file = QtWidgets.QFileDialog.getOpenFileName(self, "Select your company's logo!")
        pixmap = QPixmap('UI/icons/logo.png')
        self.logo.setPixmap(pixmap)
        # logic to update company image

    def sync_url(self):

        url = self.rtmp_url + self.uid.text()  # + '/' + self.config['DATA']['MODEL_FILE']
        # 'rtmp://www.droneitdown.com/rtmp/' + self.uid + '/' + self.config['DATA']['MODEL_FILE']
        self.url.setText(url)
        self.config['DATA']['USERNAME'] = self.uid.text()
        uid_models = [m for m in self.get_models() if self.uid.text() in m]
        if len(uid_models) == 0:
            self.stream_url = self.base_url + 'video_feed/' + self.config['DATA']['USERNAME'] + '/' + \
                              self.config['DATA']['MODEL_FILE']
        else:
            self.stream_url = self.base_url + 'video_feed/' + self.config['DATA']['USERNAME'] + '/' + \
                              natsorted(uid_models)[-1]

        if self.startstop.isChecked():
            self.startstop.click()
            self.status_2.setText('URL Changed, Press Start Button')

    def is_connected(self):
        pass

    def start_stop(self):
        if self.startstop.isChecked():
            self.startstop.setText('Stop')
            self.startstop.setStyleSheet("background-color : rgb(246, 97, 81)")
            self.start_video()

        else:
            print('trying to stop')
            self.startstop.setText('Start')
            self.startstop.setStyleSheet("background-color : rgb(28, 113, 216)")
            self.thread.stop()
            self.thread2.stop()
            if 'Failed' not in self.status_2.text():
                self.set_status('Video Stopped')

    def start_video(self):
        # r = requests.head(url)
        # if r.status_code > 205:
        #     self.set_status('Failed to Fetch Video')
        #     return False
        # create the video capture thread
        url = self.stream_url
        print(url)
        self.thread = VideoThread(url)
        self.thread2 = NameThread(url)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_time.connect(self.update_uptime)

        self.thread2.update_matched.connect(self.update_matched)
        # start the thread
        self.current_list = []
        self.matched.clear()
        self.thread.start()
        self.thread2.start()

        self.set_status('Fetching Realtime Stream')

    def set_data(self):
        # self.target_file
        # movie = QMovie('UI/icons/loader.gif')
        # self.pre_status.setMovie(movie)
        # movie.start()
        # self.pre_status.setLayout(QHBoxLayout())
        # self.pre_status.layout().addWidget(QLabel('Processing...  '))

        self.pre_status.setText('Processing....')
        if self.models.currentText() == 'Create New':
            # if self.config['DATA']['PREPROCESSING_FOLDER'] != self.folder_pre:
            if self.augment_yes.isChecked():
                aligned_folder = initialize_db(self.folder_pre, augment=True, pbar=self.progressBar)
            else:
                aligned_folder = initialize_db(self.folder_pre, augment=False, pbar=self.progressBar)

            self.config['DATA']['PREPROCESSING_FOLDER'] = self.folder_pre
            model_file = self.config['DATA']['MODEL_FILE']
            if self.uid.text() in model_file:
                model_file = model_file.split('_')
                model_file = model_file[0] + '_' + str(int(model_file[-1]) + 1)
            else:
                model_file = self.uid.text() + '_1'

            self.config['DATA']['MODEL_FILE'] = model_file
            create_embedding_dataset(os.path.join('models', model_file), aligned_folder, pbar=self.progressBar)
            self.upload_model(model_file)

            self.stream_url = self.base_url + 'video_feed/' + self.config['DATA']['USERNAME'] + '/' + \
                              self.config['DATA']['MODEL_FILE']
        else:
            self.config['DATA']['MODEL_FILE'] = self.models.currentText()

        self.current_model.setText('Current Model: ' + self.config['DATA']['MODEL_FILE'])
        self.pre_status.setText('Completed~!')
        self.set_status('Target set, press start')

    def upload_model(self, model_file):
        print(model_file)
        url = self.base_url + 'upload_model'
        file = open(os.path.join('models', model_file), 'rb')
        self.set_status('Uploading model file to server...')
        response = requests.post(url, files={"model": file}, data={"name": os.path.basename(file.name)})
        print(response.json())

    def modify(self):
        timer = QTimer(self)
        timer.timeout.connect(lambda: self.label.setText(QDateTime.currentDateTime().toString()))
        timer.start(1000)

        self.HomeButton.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page1))
        self.DataButton.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page2))
        self.SettingsButton.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page3))
        self.DIDButton.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page4))
        self.AboutButton.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page5))

        self.cimage.clicked.connect(lambda: self.updateCompanyImage())
        self.resultsButton.clicked.connect(lambda: os.system('open results'))
        self.folder.clicked.connect(lambda: self.selectPreFolder())
        #        self.timage.clicked.connect(lambda: self.selectFile())
        #        self.tvideo.clicked.connect(lambda: self.selectFile())
        # self.bulk.clicked.connect(lambda: self.selectPreFolder())
        self.data_submit.clicked.connect(lambda: self.set_data())

        self.uid.textChanged.connect(self.sync_url)
        self.startstop.clicked.connect(self.start_stop)
        #
        # validator = QRegExpValidator(QRegExp("/^[a-zA-Z]{0,7}$/"))
        # self.uid.setValidator(validator)

        self.disply_width = self.frameGeometry().width()
        self.display_height = self.frameGeometry().height()

        try:
            g = geocoder.ip('me')
            add = g.address
            g = g.latlng
            loc = f"{g[0]}, {g[1]} - {add}"
            self.loc.setText('Current Location: ' + loc)
            self.config['DATA']['LAST_LOCATION'] = loc
        except Exception as e:
            self.loc.setText('Current Location: ' + self.config['DATA']['LAST_LOCATION'])

        self.n_files.setText('Files Processed: ' + str(self.config['DATA']['FILES_PROCESSED']))
        self.n_targets.setText('Targets Matched: ' + str(self.config['DATA']['TARGETS_MATCHED']))
        self.n_faces.setText('Total Faces Detected ' + str(self.config['DATA']['FACES_DETECTED']))
        self.prefolder.setText(f'Selected Folder: {self.config["DATA"]["PREPROCESSING_FOLDER"]}')
        self.uid.setText(f'{self.config["DATA"]["USERNAME"]}')
        self.current_model.setText(f"Current Model: {self.config['DATA']['MODEL_FILE']}")
        self.current_list = []
        # Updating models dropdown
        self.models.clear()
        model_list = ['Create New'] + self.get_models()
        self.models.addItems(model_list)
        self.models.currentIndexChanged.connect(self.set_model)
        self.updateCompanyImage()
    def set_model(self):
        self.model_file = self.models.currentText()
        print(self.model_file)

    def get_models(self):
        url = self.base_url + 'get_models/' + self.uid.text()
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()['models']
            else:
                raise response.content
        except Exception as e:
            print('Failed to fetch models: ' + str(e))
            return []

    def closeEvent(self, event):
        self.set_app_config()
        if self.thread:
            self.thread.stop()
            self.thread2.stop()
        event.accept()

    @pyqtSlot(int)
    def update_uptime(self, time):
        self.uptime.setText(f"Stream Uptime: {time} seconds")

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        # print(cv_img)
        if cv_img.any():
            qt_img = self.convert_cv_qt(cv_img)
            self.image_label.setPixmap(qt_img)
        else:
            self.set_status('Failed to fetch Video (No Stream)...')
            self.startstop.click()

    @pyqtSlot(list)
    def update_matched(self, dict):
        entries = [f'{name}' for name in dict]
        for name in entries:
            if name in self.matched_dict:
                self.matched_dict[name] += 1
            else:
                self.matched_dict[name] = 1
        new = list(set(entries) - set(self.current_list))
        self.matched.addItems(new)
        self.current_list += new

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def set_user(self):
        self.user.setText(f'Welcome {self.config["DATA"]["USERNAME"]}')
        self.uid.setText(self.config['DATA']["USERNAME"])

    def set_status(self, status='Stopped'):
        self.status_2.setText(f'{status}')


def run_ui(config, url, rtmp):
    import sys

    # qdarktheme.enable_hi_dpi()
    app = QtWidgets.QApplication(sys.argv)
    qdarktheme.setup_theme()

    # mw = QtWidgets.QMainWindow()
    # ui = MainWindow()
    ex = MainWindow(config, url, rtmp)

    # timer = QTimer(ui)
    # timer.timeout.connect(showTime(ui))
    # timer.start(1000)
    ex.set_user()
    ex.set_status()
    # ui.label.setText('hi')
    #
    # ui.setupUi(MainWindow)
    # MainWindow.show()
    return app, ex.config


if __name__ == "__main__":
    run_ui()
