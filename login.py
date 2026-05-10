
import sys
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QMessageBox, QFrame
)
from PySide6.QtGui import QPixmap, QFont, QPalette, QBrush
from PySide6.QtCore import Qt

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("登录 - 人脸表情识别系统")
        self.setFixedSize(900, 650)
        
        self.main_window = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        self._set_background()
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(100, 100, 100, 100)
        
        login_frame = QFrame()
        login_frame.setFixedSize(450, 400)
        login_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.97);
                border-radius: 25px;
            }
        """)
        
        login_layout = QVBoxLayout(login_frame)
        login_layout.setContentsMargins(50, 40, 50, 40)
        login_layout.setSpacing(25)
        
        title_label = QLabel("欢迎登录")
        title_label.setFont(QFont("Microsoft YaHei", 26, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50;")
        login_layout.addWidget(title_label)
        
        username_container = QWidget()
        username_layout = QVBoxLayout(username_container)
        username_layout.setContentsMargins(0, 0, 0, 0)
        username_layout.setSpacing(8)
        
        username_label = QLabel("账号：")
        username_label.setFont(QFont("Microsoft YaHei", 12))
        username_label.setStyleSheet("color: #34495e;")
        username_label.setStyleSheet("background-color: #00000000;")
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入账号")
        self.username_input.setFont(QFont("Microsoft YaHei", 13))
        self.username_input.setMinimumHeight(45)
        self.username_input.setStyleSheet("""
            QLineEdit {
                padding: 10px 15px;
                border: 2px solid #dfe6e9;
                border-radius: 10px;
                background-color: #f8f9fa;
                color: #2c3e50;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
                background-color: white;
            }
        """)
        
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)
        login_layout.addWidget(username_container)
        
        password_container = QWidget()
        password_layout = QVBoxLayout(password_container)
        password_layout.setContentsMargins(0, 0, 0, 0)
        password_layout.setSpacing(8)
        
        password_label = QLabel("密码：")
        password_label.setFont(QFont("Microsoft YaHei", 12))
        password_label.setStyleSheet("color: #34495e;")
        password_label.setStyleSheet("background-color: #00000000;")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFont(QFont("Microsoft YaHei", 13))
        self.password_input.setMinimumHeight(45)
        self.password_input.setStyleSheet("""
            QLineEdit {
                padding: 10px 15px;
                border: 2px solid #dfe6e9;
                border-radius: 10px;
                background-color: #f8f9fa;
                color: #2c3e50;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
                background-color: white;
            }
        """)
        
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_input)
        login_layout.addWidget(password_container)
        
        self.login_button = QPushButton("登录")
        self.login_button.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        self.login_button.setMinimumHeight(50)
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.login_button.clicked.connect(self.check_login)
        login_layout.addWidget(self.login_button)
        
        login_layout.addStretch()
        
        center_layout.addWidget(login_frame, 0, Qt.AlignCenter)
        main_layout.addWidget(center_widget)
        
        self.username_input.returnPressed.connect(self.check_login)
        self.password_input.returnPressed.connect(self.check_login)
    
    def _set_background(self):
        bg_path = os.path.join(os.path.dirname(__file__), "assets", "background.jpg")
        if os.path.exists(bg_path):
            palette = QPalette()
            pixmap = QPixmap(bg_path)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            palette.setBrush(QPalette.Window, QBrush(scaled_pixmap))
            self.setPalette(palette)
        else:
            self.setStyleSheet("background-color: #34495e;")
    
    def resizeEvent(self, event):
        self._set_background()
        super().resizeEvent(event)
    
    def check_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        
        if username == "123" and password == "123":
            QMessageBox.information(self, "成功", "登录成功！")
            self.open_main_window()
        else:
            QMessageBox.warning(self, "错误", "账号或密码错误，请重新输入！")
            self.password_input.clear()
            self.password_input.setFocus()
    
    def set_main_window(self, window):
        self.main_window = window
    
    def open_main_window(self):
        if self.main_window:
            self.main_window.show()
            self.close()
