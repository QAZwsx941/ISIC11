import sys
from PyQt5 import QtWidgets, QtCore

# 得到一个应用程序,传入参数
app = QtWidgets.QApplication(sys.argv)
# 建立一个窗口
widget = QtWidgets.QWidget()
# 调节窗口大小
widget.resize(500, 500)
# 设置窗口标题
widget.setWindowTitle("Hello PyQt5!")
# 让窗口显示
widget.show()
# 执行循环
sys.exit(app.exec_())
