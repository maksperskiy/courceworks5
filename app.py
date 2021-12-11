from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import mediapipe as mp
import hands
from ui import Ui_MainWindow

app = QtWidgets.QApplication(sys.argv)

MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()

mp_h = mp.solutions.hands  # Holistic model
mp_d = mp.solutions.drawing_utils  # Drawing utilities
mp_ds = mp.solutions.drawing_styles

ui.textEdit.setText("default")

get_actions = lambda text: text.split("\n")


def demo():
    hands.only_hands(mp_h, mp_d, mp_ds, camera=int(ui.spinBox.text()))


def start():
    hands.make_dirs(os.path.join(ui.lineEdit.text()),
                    actions=get_actions(ui.textEdit.toPlainText()),
                    no_sequences=int(ui.lineEdit_2.text()))
    hands.get_keypoints(mp_h, mp_d, mp_ds,
                        get_actions(ui.textEdit.toPlainText()),
                        no_sequences=int(ui.lineEdit_2.text()),
                        sequence_length=int(ui.lineEdit_3.text()),
                        data_path=os.path.join(ui.lineEdit.text()),
                        camera=int(ui.spinBox.text()))


ui.pushButton.clicked.connect(start)
ui.pushButton_2.clicked.connect(demo)

sys.exit(app.exec_())
