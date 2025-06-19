import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from ui_mainwindow import Ui_MainWindow
from registration_pcd import RegistrationPCD
import os

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.setWindowTitle("点云配准软件")
        
        # 连接信号和槽
        self.btnSelectPCD.clicked.connect(self.select_pcd_folder)
        self.btnSelectGPS.clicked.connect(self.select_gps_file)
        self.btnSelectSave.clicked.connect(self.select_save_folder)
        self.btnRegister.clicked.connect(self.start_registration)
        
        # 初始化变量
        self.pcd_folder = ""
        self.gps_file = ""
        self.save_folder = ""
        
        # 设置默认参数
        self.spinMinZ.setValue(-1.0)
        self.spinMaxZ.setValue(20.0)
        self.spinVoxel.setValue(0.3)
    
    def select_pcd_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择PCD文件夹")
        if folder:
            self.pcd_folder = folder
            self.labelPCDPath.setText(f"已选择: {folder}")
            
            pcd_files = [f for f in os.listdir(folder) if f.endswith('.pcd')]
            if not pcd_files:
                QMessageBox.warning(self, "警告", "所选文件夹中没有找到.pcd文件!")
            else:
                self.labelPCDCount.setText(f"找到 {len(pcd_files)} 个PCD文件")
    
    def select_gps_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择GPS文件", "", "文本文件 (*.txt)")
        if file:
            self.gps_file = file
            self.labelGPSPath.setText(f"已选择: {file}")
            
            try:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < 1:
                        QMessageBox.warning(self, "警告", "GPS文件为空!")
                    else:
                        self.labelGPSCount.setText(f"找到 {len(lines)} 行数据")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法读取GPS文件: {str(e)}")
    
    def select_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择结果保存文件夹")
        if folder:
            self.save_folder = folder
            self.labelSavePath.setText(f"已选择: {folder}")
    
    def start_registration(self):
        if not self.pcd_folder:
            QMessageBox.warning(self, "警告", "请先选择PCD文件夹!")
            return
            
        if not self.gps_file:
            QMessageBox.warning(self, "警告", "请先选择GPS文件!")
            return
            
        if not self.save_folder:
            QMessageBox.warning(self, "警告", "请先选择结果保存文件夹!")
            return
            
        z_min = self.spinMinZ.value()
        z_max = self.spinMaxZ.value()
        voxel_size = self.spinVoxel.value()
        
        if z_min >= z_max:
            QMessageBox.warning(self, "警告", "最小高程必须小于最大高程!")
            return
            
        try:
            # 验证输入
            if not all([self.pcd_folder, self.gps_file, self.save_folder]):
                raise ValueError("请完成所有必选项选择")
        
            self.labelStatus.setText("正在配准...")
            QApplication.processEvents()

            # 执行配准
            data = {
                'pcd_folder': self.pcd_folder,
                'gps_file': self.gps_file,
                'save_folder': self.save_folder,
                'z_min': self.spinMinZ.value(),
                'z_max': self.spinMaxZ.value(),
                'voxel_size': self.spinVoxel.value()
            }
            
            rp = RegistrationPCD(data)
            registered_pcd = rp.registration()  # 此步骤可能抛出未捕获的异常

            # 显式验证结果文件
            if not os.path.exists(rp.save_path):
                raise FileNotFoundError("结果文件未生成")

            # 更新UI状态
            self.labelStatus.setText("配准成功")
            QMessageBox.information(self, "成功", f"结果已保存到:\n{rp.save_path}")

            # 可视化结果
            rp.visualize(registered_pcd)

        except Exception as e:
            self.labelStatus.setText("配准失败")
            QMessageBox.critical(self, "错误", f"错误详情:\n{str(e)}")
            #print("DEBUG - 错误堆栈:", traceback.format_exc())  # 打印完整堆栈

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())