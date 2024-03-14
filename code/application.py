from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QDateTime, Qt
from PyQt5 import uic

import sys
import numpy as np
import pandas as pd

import feature_extractor as fe
import emotion_analysis_model as eam
import convert_signal_to_music as csm


form_application_class = uic.loadUiType("../GUI/main.ui")[0]


class Menu(QMainWindow, form_application_class):
    def __init__(self, parent=None):
        super(Menu, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Brain To Music")
        self.setWindowIcon(QIcon('../GUI/Brain_with_art.jpg'))
        self.setFixedSize(self.size())
        self.setAcceptDrops(True)
        self.Arousal.setStyleSheet("Color : red")
        self.Valence.setStyleSheet("Color : red")

        self.count = 0
        self.eeg_file_path = ""
        self.total_feature = pd.DataFrame([])
        self.beta_gamma_max_freq_list = []
        self.time_frequency_list = []

        self.listWidget.setAcceptDrops(True)
        self.extract_feature.clicked.connect(self.extract_feature_event)
        self.emotion_analysis.clicked.connect(self.emotion_analysis_event)
        self.convert_music.clicked.connect(self.convert_music_event)
        self.reset_button.clicked.connect(self.reset_event)

        self.extract_feature.setStyleSheet(
            "QPushButton{ color: white; background-color: rgb(58, 73, 80); border-radius: 15px;}")
        self.emotion_analysis.setStyleSheet(
            "QPushButton{ color: white; background-color: rgb(58, 73, 80); border-radius: 15px;}")
        self.convert_music.setStyleSheet(
            "QPushButton{ color: white; background-color: rgb(58, 73, 80); border-radius: 15px;}")
        self.reset_button.setStyleSheet(
            "QPushButton{ color: white; background-color: rgb(58, 73, 80); border-radius: 15px;}")

        self.index_channel = ["FP1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7",
                              "PO3", "O1", "Oz", "Pz", "FP2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
                              "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"]
        self.basic_channel = ["F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "FP1", "FP2", "AF3", "AF4"]

        self.mean_list = [self.Mean_1, self.Mean_2, self.Mean_3, self.Mean_4, self.Mean_5, self.Mean_6,
                          self.Mean_7, self.Mean_8, self.Mean_9, self.Mean_10, self.Mean_11, self.Mean_12]
        self.std_list = [self.Std_1, self.Std_2, self.Std_3, self.Std_4, self.Std_5, self.Std_6,
                         self.Std_7, self.Std_8, self.Std_9, self.Std_10, self.Std_11, self.Std_12]
        self.skewness_list = [self.Skewness_1, self.Skewness_2, self.Skewness_3, self.Skewness_4, self.Skewness_5, self.Skewness_6,
                              self.Skewness_7, self.Skewness_8, self.Skewness_9, self.Skewness_10, self.Skewness_11, self.Skewness_12]
        self.kurotosis_list = [self.Kurotosis_1, self.Kurotosis_2, self.Kurotosis_3, self.Kurotosis_4, self.Kurotosis_5, self.Kurotosis_6,
                               self.Kurotosis_7, self.Kurotosis_8, self.Kurotosis_9, self.Kurotosis_10, self.Kurotosis_11, self.Kurotosis_12]
        self.hjorth_mob_list = [self.Hjorth_Mobility_1, self.Hjorth_Mobility_2, self.Hjorth_Mobility_3, self.Hjorth_Mobility_4, self.Hjorth_Mobility_5, self.Hjorth_Mobility_6,
                                self.Hjorth_Mobility_7, self.Hjorth_Mobility_8, self.Hjorth_Mobility_9, self.Hjorth_Mobility_10, self.Hjorth_Mobility_11, self.Hjorth_Mobility_12]
        self.hjorth_com_list = [self.Hjorth_Complexity_1, self.Hjorth_Complexity_2, self.Hjorth_Complexity_3, self.Hjorth_Complexity_4, self.Hjorth_Complexity_5, self.Hjorth_Complexity_6,
                               self.Hjorth_Complexity_7, self.Hjorth_Complexity_8, self.Hjorth_Complexity_9, self.Hjorth_Complexity_10, self.Hjorth_Complexity_11, self.Hjorth_Complexity_12]
        self.hurst_list = [self.Hurst_Exponent_1, self.Hurst_Exponent_2, self.Hurst_Exponent_3, self.Hurst_Exponent_4, self.Hurst_Exponent_5, self.Hurst_Exponent_6,
                           self.Hurst_Exponent_7, self.Hurst_Exponent_8, self.Hurst_Exponent_9, self.Hurst_Exponent_10, self.Hurst_Exponent_11, self.Hurst_Exponent_12]
        self.dfa_list = [self.DFA_1, self.DFA_2, self.DFA_3, self.DFA_4, self.DFA_5, self.DFA_6,
                         self.DFA_7, self.DFA_8, self.DFA_9, self.DFA_10, self.DFA_11, self.DFA_12]
        self.hfd_list = [self.Higuchi_FD_1, self.Higuchi_FD_2, self.Higuchi_FD_3, self.Higuchi_FD_4, self.Higuchi_FD_5, self.Higuchi_FD_6,
                         self.Higuchi_FD_7, self.Higuchi_FD_8, self.Higuchi_FD_9, self.Higuchi_FD_10, self.Higuchi_FD_11, self.Higuchi_FD_12]
        self.pfd_list = [self.Petrosian_FD_1, self.Petrosian_FD_2, self.Petrosian_FD_3, self.Petrosian_FD_4, self.Petrosian_FD_5, self.Petrosian_FD_6,
                         self.Petrosian_FD_7, self.Petrosian_FD_8, self.Petrosian_FD_9, self.Petrosian_FD_10, self.Petrosian_FD_11, self.Petrosian_FD_12]
        self.max_psd_list = [self.PSD_1, self.PSD_2, self.PSD_3, self.PSD_4, self.PSD_5, self.PSD_6,
                             self.PSD_7, self.PSD_8, self.PSD_9, self.PSD_10, self.PSD_11, self.PSD_12]
        self.max_freq_list = [self.Maximum_PSF_1, self.Maximum_PSF_2, self.Maximum_PSF_3, self.Maximum_PSF_4, self.Maximum_PSF_5, self.Maximum_PSF_6,
                              self.Maximum_PSF_7, self.Maximum_PSF_8, self.Maximum_PSF_9, self.Maximum_PSF_10, self.Maximum_PSF_11, self.Maximum_PSF_12]
        self.rms_list = [self.RMS_1, self.RMS_2, self.RMS_3, self.RMS_4, self.RMS_5, self.RMS_6,
                         self.RMS_7, self.RMS_8, self.RMS_9, self.RMS_10, self.RMS_11, self.RMS_12]
        self.feature_names = ["mean", "std", "skewness", "kurotosis", "hjorth_mob", "hjorth_com", "hurst", "dfa", "hfd", "pfd", "MAX_PSD", "MAX_freq", "RMS"]
        self.feature_lists = [self.mean_list, self.std_list, self.skewness_list, self.kurotosis_list, self.hjorth_mob_list, self.hjorth_com_list,
                              self.hurst_list, self.dfa_list, self.hfd_list, self.pfd_list, self.max_psd_list, self.max_freq_list, self.rms_list]

    def reset(self):
        self.count = 0
        self.eeg_file_path = ""
        self.total_feature = pd.DataFrame([])
        self.beta_gamma_max_freq_list = []
        self.time_frequency_list = []

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.listWidget.clear()
        if event.mimeData().hasUrls() and self.count == 0:
            event.setDropAction(Qt.CopyAction)
            event.accept()

            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = str(url.toLocalFile())
                file = file_path.split('/')[-1]
            else:
                file_path = str(url.toString())

            self.listWidget.addItem(file)
            self.count += 1
            self.eeg_file_path = file_path

        else:
            event.ignore()

    def extract_feature_event(self):
        if self.count == 0 or self.listWidget.item(0).text()[-4:] != ".csv":
            self.reset()
            self.listWidget.clear()
            self.listWidget.addItem("!! Import EEG signal first !!")

        else:
            if self.total_feature.shape != (0, 0):
                return

            # basic_channel 중 첫 3초 제거
            self.eeg_file = pd.read_csv(self.eeg_file_path, header=None)
            self.eeg_file.index = self.index_channel
            self.eeg_file = self.eeg_file.loc[self.basic_channel, 128 * 3:]

            # feature 추출
            time_feature = fe.time_feature_extractor(self.eeg_file)
            self.listWidget.addItem("Time Feature .... Done ✨")
            frequency_feature = fe.frequency_feature_extractor(self.eeg_file)
            frequency_feature.index = self.basic_channel
            self.listWidget.addItem("Frequency Feature .... Done ✨")
            self.beta_gamma_max_freq_list, self.time_frequency_list = fe.time_frequency_feature_extractor(self.eeg_file)
            self.listWidget.addItem("Time Frequency Feature .... Done ✨")

            self.total_feature = pd.concat([time_feature, frequency_feature], axis=1)
            self.total_feature["label"] = np.full(self.total_feature.shape[0], -1)

            # display
            for idx, channel in enumerate(self.basic_channel):
                for feature_list, feature_name in zip(self.feature_lists, self.feature_names):
                    feature_list[idx].setText(str(round(self.total_feature.loc[channel, feature_name], 4)))

            self.listWidget.addItem("")

    def emotion_analysis_event(self):
        if self.total_feature.shape == (0, 0):
            self.reset()
            self.listWidget.clear()
            self.listWidget.addItem("!! Extract Features First !!")

        else:
            self.arousal_model = eam.load_model("Arousal_0.8")
            self.valence_model = eam.load_model("Valence_0.8")
            self.listWidget.addItem("Load Model .... Done ✨")

            self.arousal_predict = eam.predict_emotion(self.total_feature, self.arousal_model)
            self.valence_predict = eam.predict_emotion(self.total_feature, self.valence_model)

            if self.arousal_predict.index[0] == 1:
                self.Arousal.setText("High")
            elif self.arousal_predict.index[0] == 0:
                self.Arousal.setText("Low")

            if self.valence_predict.index[0] == 1:
                self.Valence.setText("High")
            elif self.valence_predict.index[0] == 0:
                self.Valence.setText("Low")

            self.listWidget.addItem("Emotion Analysis .... Done ✨")
            self.listWidget.addItem("")

    def convert_music_event(self):
        if len(self.time_frequency_list) == 0:
            self.reset()
            self.listWidget.clear()
            self.listWidget.addItem("!! Extract Features First !!")

        else:
            inst = self.select_inst.currentText()
            scale_name_list, music = csm.convert_signal_to_music(inst, self.eeg_file, self.arousal_predict, self.valence_predict, self.time_frequency_list, self.beta_gamma_max_freq_list)
            music.write(f"../music/{self.eeg_file_path.split('/')[-1][:-4]}_{inst}.mid")

            string = f"{scale_name_list[0]} -> {scale_name_list[1]}"
            self.listWidget.addItem(f"{string} chord .... ")
            self.listWidget.addItem(f"                              .... Complete 🎧")
            self.listWidget.addItem("")

    def reset_event(self):
        self.reset()
        self.listWidget.clear()
        for feature_list in self.feature_lists:
            for item in feature_list:
                item.clear()
        self.Arousal.clear()
        self.Valence.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    menu = Menu()
    menu.show()
    sys.exit(app.exec_())
