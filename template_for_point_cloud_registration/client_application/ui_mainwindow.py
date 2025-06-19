from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # 设置主布局
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(15, 15, 15, 15)
        self.verticalLayout.setSpacing(15)
        
        # 1. 添加LOGO和标题栏（居中）
        self.setup_header()
        
        # 2. 功能区域
        self.setup_pcd_section()
        self.setup_gps_section()
        self.setup_save_section()
        self.setup_parameters()
        
        # 3. 操作按钮区域
        self.setup_action_buttons()
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # 应用样式
        self.setup_styles()

    def setup_header(self):
        """添加LOGO和标题（居中）"""
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        
        # 添加左侧弹性空间
        header.addStretch()
        
        # 公司LOGO
        self.logo_label = QtWidgets.QLabel(self.centralwidget)
        self.logo_label.setObjectName("logo_label")
        self.logo_label.setPixmap(QtGui.QPixmap("assets/optosky_logo.png").scaled(
            120, 40, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        header.addWidget(self.logo_label)
        
        # 标题（与LOGO保持间距）
        self.title_label = QtWidgets.QLabel(self.centralwidget)
        self.title_label.setObjectName("title_label")
        header.addWidget(self.title_label)
        
        # 添加右侧弹性空间
        header.addStretch()
        
        self.verticalLayout.addLayout(header)

    def setup_pcd_section(self):
        """PCD文件夹选择区域"""
        self.groupPCD = QtWidgets.QGroupBox(self.centralwidget)
        self.groupPCD.setObjectName("groupPCD")
        
        layout = QtWidgets.QVBoxLayout(self.groupPCD)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(10)
        
        # 选择按钮
        self.btnSelectPCD = QtWidgets.QPushButton(self.groupPCD)
        self.btnSelectPCD.setObjectName("btnSelectPCD")
        self.btnSelectPCD.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        layout.addWidget(self.btnSelectPCD)
        
        # 路径显示
        self.labelPCDPath = QtWidgets.QLabel(self.groupPCD)
        self.labelPCDPath.setObjectName("labelPCDPath")
        self.labelPCDPath.setWordWrap(True)
        layout.addWidget(self.labelPCDPath)
        
        # 文件计数
        self.labelPCDCount = QtWidgets.QLabel(self.groupPCD)
        self.labelPCDCount.setObjectName("labelPCDCount")
        layout.addWidget(self.labelPCDCount)
        
        self.verticalLayout.addWidget(self.groupPCD)

    def setup_gps_section(self):
        """GPS文件选择区域"""
        self.groupGPS = QtWidgets.QGroupBox(self.centralwidget)
        self.groupGPS.setObjectName("groupGPS")
        
        layout = QtWidgets.QVBoxLayout(self.groupGPS)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(10)
        
        self.btnSelectGPS = QtWidgets.QPushButton(self.groupGPS)
        self.btnSelectGPS.setObjectName("btnSelectGPS")
        self.btnSelectGPS.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        layout.addWidget(self.btnSelectGPS)
        
        self.labelGPSPath = QtWidgets.QLabel(self.groupGPS)
        self.labelGPSPath.setObjectName("labelGPSPath")
        self.labelGPSPath.setWordWrap(True)
        layout.addWidget(self.labelGPSPath)
        
        self.labelGPSCount = QtWidgets.QLabel(self.groupGPS)
        self.labelGPSCount.setObjectName("labelGPSCount")
        layout.addWidget(self.labelGPSCount)
        
        self.verticalLayout.addWidget(self.groupGPS)

    def setup_save_section(self):
        """保存文件夹选择区域"""
        self.groupSave = QtWidgets.QGroupBox(self.centralwidget)
        self.groupSave.setObjectName("groupSave")
        
        layout = QtWidgets.QVBoxLayout(self.groupSave)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(10)
        
        self.btnSelectSave = QtWidgets.QPushButton(self.groupSave)
        self.btnSelectSave.setObjectName("btnSelectSave")
        self.btnSelectSave.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        layout.addWidget(self.btnSelectSave)
        
        self.labelSavePath = QtWidgets.QLabel(self.groupSave)
        self.labelSavePath.setObjectName("labelSavePath")
        self.labelSavePath.setWordWrap(True)
        layout.addWidget(self.labelSavePath)
        
        self.verticalLayout.addWidget(self.groupSave)

    def setup_parameters(self):
        """参数设置区域"""
        self.groupParams = QtWidgets.QGroupBox(self.centralwidget)
        self.groupParams.setObjectName("groupParams")
        
        self.gridLayout = QtWidgets.QGridLayout(self.groupParams)
        self.gridLayout.setContentsMargins(10, 15, 10, 10)
        self.gridLayout.setHorizontalSpacing(20)
        self.gridLayout.setVerticalSpacing(10)
        
        # 最小高程
        self.labelMinZ = QtWidgets.QLabel(self.groupParams)
        self.labelMinZ.setObjectName("labelMinZ")
        self.gridLayout.addWidget(self.labelMinZ, 0, 0, 1, 1)
        
        self.spinMinZ = QtWidgets.QDoubleSpinBox(self.groupParams)
        self.spinMinZ.setObjectName("spinMinZ")
        self.spinMinZ.setMinimum(-9999.99)
        self.spinMinZ.setMaximum(9999.99)
        self.spinMinZ.setSingleStep(0.1)
        self.gridLayout.addWidget(self.spinMinZ, 0, 1, 1, 1)
        
        # 最大高程
        self.labelMaxZ = QtWidgets.QLabel(self.groupParams)
        self.labelMaxZ.setObjectName("labelMaxZ")
        self.gridLayout.addWidget(self.labelMaxZ, 1, 0, 1, 1)
        
        self.spinMaxZ = QtWidgets.QDoubleSpinBox(self.groupParams)
        self.spinMaxZ.setObjectName("spinMaxZ")
        self.spinMaxZ.setMinimum(-9999.99)
        self.spinMaxZ.setMaximum(9999.99)
        self.spinMaxZ.setSingleStep(0.1)
        self.gridLayout.addWidget(self.spinMaxZ, 1, 1, 1, 1)
        
        # 体素大小
        self.labelVoxel = QtWidgets.QLabel(self.groupParams)
        self.labelVoxel.setObjectName("labelVoxel")
        self.gridLayout.addWidget(self.labelVoxel, 2, 0, 1, 1)
        
        self.spinVoxel = QtWidgets.QDoubleSpinBox(self.groupParams)
        self.spinVoxel.setObjectName("spinVoxel")
        self.spinVoxel.setMinimum(0.01)
        self.spinVoxel.setMaximum(10.0)
        self.spinVoxel.setSingleStep(0.01)
        self.gridLayout.addWidget(self.spinVoxel, 2, 1, 1, 1)
        
        self.verticalLayout.addWidget(self.groupParams)

    def setup_action_buttons(self):
        """操作按钮区域"""
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(20)
        
        # 配准按钮
        self.btnRegister = QtWidgets.QPushButton(self.centralwidget)
        self.btnRegister.setObjectName("btnRegister")
        self.btnRegister.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn_layout.addWidget(self.btnRegister)
        
        # 状态显示
        self.labelStatus = QtWidgets.QLabel(self.centralwidget)
        self.labelStatus.setObjectName("labelStatus")
        self.labelStatus.setAlignment(QtCore.Qt.AlignCenter)
        btn_layout.addWidget(self.labelStatus)
        
        self.verticalLayout.addLayout(btn_layout)

    def setup_styles(self):
        """设置全局样式"""
        self.centralwidget.setStyleSheet("""
            /* 全局样式 */
            QWidget {
                font-family: 'Microsoft YaHei';
                font-size: 10pt;
            }
            
            /* 标题样式 */
            #title_label {
                font-size: 18px;
                font-weight: bold;
                color: #2A5C9A;
                margin-left: 10px;
            }
            
            /* LOGO样式 */
            #logo_label {
                margin-right: 10px;
            }
            
            /* 分组框样式 */
            QGroupBox {
                border: 1px solid #D0D0D0;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #2A5C9A;
                font-weight: bold;
            }
            
            /* 按钮样式 */
            QPushButton {
                background-color: #3A7CB9;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4B8DCA;
            }
            QPushButton:pressed {
                background-color: #2A5C9A;
            }
            #btnRegister {
                min-width: 150px;
                padding: 10px;
                font-size: 12pt;
            }
            
            /* 输入框样式 */
            QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #D0D0D0;
                border-radius: 3px;
                background-color: #FFFFFF;
            }
            
            /* 标签样式 */
            QLabel {
                color: #555555;
            }
            #labelStatus {
                font-weight: bold;
                color: #2A5C9A;
            }
        """)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "点云配准软件"))
        self.title_label.setText(_translate("MainWindow", "点云配准软件"))
        
        # 分组框标题
        self.groupPCD.setTitle(_translate("MainWindow", "1. 选择PCD文件夹"))
        self.groupGPS.setTitle(_translate("MainWindow", "2. 选择GPS文件"))
        self.groupSave.setTitle(_translate("MainWindow", "3. 选择结果保存文件夹"))
        self.groupParams.setTitle(_translate("MainWindow", "4. 设置配准参数"))
        
        # 按钮文本
        self.btnSelectPCD.setText(_translate("MainWindow", "选择PCD文件夹"))
        self.btnSelectGPS.setText(_translate("MainWindow", "选择GPS文件"))
        self.btnSelectSave.setText(_translate("MainWindow", "选择保存文件夹"))
        self.btnRegister.setText(_translate("MainWindow", "开始配准"))
        
        # 标签文本
        self.labelPCDPath.setText(_translate("MainWindow", "未选择"))
        self.labelPCDCount.setText(_translate("MainWindow", "未扫描"))
        self.labelGPSPath.setText(_translate("MainWindow", "未选择"))
        self.labelGPSCount.setText(_translate("MainWindow", "未扫描"))
        self.labelSavePath.setText(_translate("MainWindow", "未选择"))
        self.labelMinZ.setText(_translate("MainWindow", "最小高程:"))
        self.labelMaxZ.setText(_translate("MainWindow", "最大高程:"))
        self.labelVoxel.setText(_translate("MainWindow", "体素大小:"))
        self.labelStatus.setText(_translate("MainWindow", "准备就绪"))