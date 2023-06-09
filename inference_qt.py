# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'inference_qt.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from PIL import Image

import torch
from torchvision import transforms as T
import random
from network import R2AttU_Net


model_path = "./models/R2AttU_Net-200-0.0003-108-0.4001.pkl"

def tensor2img(x):
    img = x.detach().numpy()
    img = img*255
    #img = img *127
    img = img.astype('uint8')
    return img

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(933, 697)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(80, 180, 241, 271))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(500, 180, 241, 271))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(130, 120, 161, 41))
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(350, 220, 121, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(350, 310, 121, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(540, 120, 161, 41))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(260, 20, 301, 71))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 933, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(MainWindow.btn_showImage) # type: ignore
        self.graphicsView.rubberBandChanged['QRect','QPointF','QPointF'].connect(MainWindow.btn_showImage) # type: ignore
        self.pushButton_2.clicked.connect(MainWindow.btn_showResult) # type: ignore
        self.graphicsView_2.rubberBandChanged['QRect','QPointF','QPointF'].connect(MainWindow.btn_showResult) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">输入图片</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "载入图片"))
        self.pushButton_2.setText(_translate("MainWindow", "分析结果"))
        self.textBrowser_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">输出结果</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "分割任务识别界面"))

    def btn_showImage(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "./", "png (*.png)")
        self.file_name = file_name[0]
        image_path = file_name[0]
        if (file_name[0] == ""):
            QtWidgets.QMessageBox.information(self, "提示", self.tr("没有选择图片文件！"))
        print(image_path)
        img = cv2.imread(image_path)  # 读取图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        img = cv2.resize(img, dsize=(200, 200))
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        self.zoomscale = 1  # 图片放缩尺度
        frame = QtGui.QImage(img, x, y, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(frame)
        self.item = QtWidgets.QGraphicsPixmapItem(pix)  # 创建像素图元
        self.scene = QtWidgets.QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.show()


    def btn_showResult(self):
        model = R2AttU_Net(img_ch=3, output_ch=1, t=3)
        # model = R2AttU_Net(img_ch=3, output_ch=3, t=3)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # 2. 导入图像
        image = Image.open(self.file_name)
        image = image.convert('RGB')

        aspect_ratio = image.size[1] / image.size[0]

        Transform = []

        ResizeRange = random.randint(300, 320)
        Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))
        p_transform = random.random()

        Transform.append(T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)

        input = image.unsqueeze(dim=0)
        output = torch.sigmoid(model(input)).squeeze()

        out_image = tensor2img(output)
        # out_image = np.where(out_image >= 35, 1, 0)
        # out_image = out_image*255
        # out_pil = Image.fromarray(out_image)

        img = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)  # 转换图像通道
        img = cv2.resize(img, dsize=(200, 200))
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        self.zoomscale = 1  # 图片放缩尺度
        frame = QtGui.QImage(img, x, y, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(frame)
        self.item = QtWidgets.QGraphicsPixmapItem(pix)  # 创建像素图元
        self.scene = QtWidgets.QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsView_2.setScene(self.scene)
        self.graphicsView_2.show()