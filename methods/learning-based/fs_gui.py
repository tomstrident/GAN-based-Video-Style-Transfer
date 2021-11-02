# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:06:47 2020

@author: Tom
"""

import cv2
import sys
from PyQt5 import QtCore
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QSlider, QPushButton, QWidget, QVBoxLayout, QApplication, QGridLayout, QGroupBox, QLabel, QFileDialog 

import os
import imageio
import numpy as np
#from fs_johnson import Johnson
from fs_huang import Huang

class App(QWidget):
  def __init__(self):
    super().__init__()
    self.title = "Optimal Fast Style Transfer Demo"
    self.image_folder = os.getcwd() + "/styles/"
    self.grid = QGridLayout()
    self.initUI()
    
    self.cap = None
    self.src = "vsttest.mp4"
    
    self.video_output = False
  
  def getInputSources(self):
    index = 0
    arr = []
    while True:
      cap = cv2.VideoCapture(index)
      if not cap.read()[0]:
        break
      else:
        arr.append(index)
      cap.release()
      index += 1
    return arr
  
  def addToGrid(self, name, wgts, x, y):
    groupBox = QGroupBox(name)
    vbox = QVBoxLayout()
    for wgt in wgts:
      vbox.addWidget(wgt)
    groupBox.setLayout(vbox)
    self.grid.addWidget(groupBox, x, y)
  
  def loadTorchFile(self, file_name):
    splits = file_name.split('/')
    
    if len(splits) < 4:
      print("something is not right ..")
      return
    
    dset, method, run_id, file = splits[-4:]
    print(splits[-4:])
    
    if method == "huang":
      self.model = Huang()
    else:
      print("invalid model path")
      return
    
    if run_id[0] == 'm':
      n_styles = int(run_id[4])
    else:
      n_styles = int(run_id[3])
      
    print("n_styles", n_styles)
    
    self.model.loadModelID(n_styles, file_name)
    self.style_id = 0
  
  @QtCore.pyqtSlot()
  def startWebcam(self):
    self.timer = QtCore.QTimer(self, interval=40)
    self.timer.timeout.connect(self.updateFrame)
    
    #if self.cap is None:
    self.cap = cv2.VideoCapture(self.src)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    self.timer.start()
    self.frame_counter = 0
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    self.save_snap.setEnabled(False)
    self.snp_btn.setEnabled(True)
    
  @QtCore.pyqtSlot()
  def updateFrame(self):
    ret, image = self.cap.read()
    self.frame_counter += 1
    
    if self.frame_counter == self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
      self.frame_counter = 0
      self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    #image = cv2.flip(image, 1)
    image = image[:,:,[2, 1, 0]]/255.0
    self.styled_image = self.model.styleFrame(image, self.style_id)
    self.styled_image = self.styled_image[:,:,[2, 1, 0]]*255.0
    self.styled_image = np.clip(self.styled_image, 0, 255).astype('uint8')
        
    #convertToQtFormat = QImage(styled.tobytes(), w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
    #p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
    #self.changePixmap.emit(p)
    
    self.displayImage(self.styled_image, True)
  
  def displayImage(self, img, window=True):
    qformat = QImage.Format_Indexed8
    if len(img.shape)==3 :
      if img.shape[2]==4:
        qformat = QImage.Format_RGBA8888
      else:
        qformat = QImage.Format_RGB888
    outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
    outImage = outImage.rgbSwapped()
    if window:
      self.label.setPixmap(QPixmap.fromImage(outImage))
  
  def toggleVideo(self):
    if self.video_output:
      self.video_output = False
      #self.pre_btn.setText("Start")
      self.save_snap.setEnabled(True)
      self.timer.stop()
    else:
      self.video_output = True
      self.startWebcam()
  
  def openFileNameDialog(self):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","PyTorch Files (*.pth)", options=options)
    if not fileName:
      return
    
    print(fileName)
    
    splits = fileName.split('.')
    
    if splits[-1] != "pth":
      print("invalid file name")
      return
    
    if self.video_output:
      self.timer.stop()
      
    self.video_output = True
    self.loadTorchFile(fileName)
    self.startWebcam()
  
  def saveFileDialog(self):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getSaveFileName(self,"Save Snapshot","","JPEG image (*.jpeg *.jpg);;PNG image (*.png)", options=options)
    
    if not fileName:
      return
    
    print(fileName)
    splits = fileName.split('.')
    
    if len(splits) > 2:
      print("invalid file name")
      return
      
    if splits[-1] != "jpeg" or splits[-1] != "jpg" or splits[-1] != "png":
      fileName = splits[0] + ".jpg"
      
    imageio.imwrite(fileName, self.styled_image[:,:,[2, 1, 0]])
  
  def setStyle(self, sid):
    self.style_id = sid
    self.slider_ref.setValue(sid*10)
  
  def addStyleButton(self, path, sid):
    button = QPushButton('', self)
    button.clicked.connect(lambda:self.setStyle(sid))
    button.setIcon(QIcon(path))
    button.setIconSize(QSize(100, 100))
    return button
  
  def selectionChange(self, i):
    if self.cb.currentText() != "vsttest.mp4":
      self.src = int(self.cb.currentText())
    else:
      self.src = self.cb.currentText()
    
    if self.video_output:
      self.timer.stop()
      self.startWebcam()
  
  def dummyPrint(self, data):
    #print(data/10)
    self.style_id = data/10
  
  def addSlider(self, name, val_start, val_min, val_max, lmin, lmax, label1, label2):
    slideBox = QGroupBox(name)
    sbox = QVBoxLayout()

    slider_vbox = QVBoxLayout()
    slider_hbox = QHBoxLayout()
    
    slider_hbox.setContentsMargins(0, 0, 0, 0)
    slider_vbox.setContentsMargins(0, 0, 0, 0)
    slider_vbox.setSpacing(0)
    
    lbox = QHBoxLayout()
    lbox.addWidget(QLabel(label1, alignment=Qt.AlignLeft))
    lbox.addWidget(QLabel(label2, alignment=Qt.AlignRight))
    
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(val_min)
    slider.setMaximum(val_max)
    slider.setValue(val_start)
    slider.setTickPosition(QSlider.TicksRight)
    slider.setTickInterval(1)
    slider.setSingleStep(1)
    
    slider.valueChanged.connect(lambda:self.dummyPrint(slider.value()))
    
    self.slider_ref = slider

    slider_vbox.addWidget(slider)
    #slider_vbox.addLayout(slider_hbox)
    
    label_min = QLabel(lmin, alignment=Qt.AlignLeft)#str(val_min)
    #label_cnt = QLabel("1", alignment=Qt.AlignLeft)#str(val_min)
    label_max = QLabel(lmax, alignment=Qt.AlignRight)#str(val_max)
    slider_hbox.addWidget(label_min, Qt.AlignLeft)
    #slider_hbox.addWidget(label_cnt, Qt.AlignCenter)
    slider_hbox.addWidget(label_max, Qt.AlignRight)
    slider_vbox.addStretch()
    
    sbox.addLayout(lbox)
    sbox.addLayout(slider_vbox)
    sbox.addLayout(slider_hbox)
    
    slideBox.setLayout(sbox)
    
    return slideBox
  
  def quitApp(self):
    if self.video_output:
      self.timer.stop()
    self.close()
    print("stopped")
  
  def initUI(self):
    self.grid = QGridLayout()
    self.setWindowTitle(self.title)
    #self.setGeometry(self.left, self.top, self.width, self.height)
    #self.resize(800, 550)
    
    #cam
    self.label = QLabel(self)
    self.addToGrid("Style Cam", [self.label], 0, 0)
    
    #style buttons
    s1 = self.addStyleButton(self.image_folder + "autoportrait.jpg", 0)
    s2 = self.addStyleButton(self.image_folder + "edtaonisl.jpg", 1)
    s3 = self.addStyleButton(self.image_folder + "composition.jpg", 2)
    self.addToGrid("Styles", [s1, s2, s3], 0, 1)
    
    #sliders
    #self.style_slider = self.addSlider("", 0, -10, 10, "1000", "1000", "Style", "Content", True)
    self.style_slider = self.addSlider("Style Slider", 0, 0, 20, "0", "2", "", "")
    self.addToGrid("Configuration", [self.style_slider], 1, 0)
    
    #buttons
    self.select_btn = QPushButton("Select", self)
    self.select_btn.clicked.connect(self.openFileNameDialog)
    
    self.snp_btn = QPushButton("Snapshot", self)
    self.snp_btn.clicked.connect(self.toggleVideo)
    self.snp_btn.setEnabled(False)
    
    self.save_snap = QPushButton("Save Snapshot", self)
    self.save_snap.clicked.connect(self.saveFileDialog)
    self.save_snap.setEnabled(False)
    
    self.exit_btn = QPushButton("Exit", self)
    self.exit_btn.clicked.connect(self.quitApp)
    
    #combo box
    src_list = self.getInputSources()
    self.cb = QComboBox()
    self.cb.addItem("vsttest.mp4")
    for src in src_list:
      self.cb.addItem(str(src))
    self.cb.currentIndexChanged.connect(self.selectionChange)
    
    self.addToGrid("", [self.select_btn, self.cb, self.snp_btn, self.save_snap, self.exit_btn], 1, 1)

    self.setLayout(self.grid)

    #self.setGeometry(100,100,200,200)
    self.show()

if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = App()
  sys.exit(app.exec_())