
import sys
import os
import cv2
import requests
import json
import markdown
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QScrollArea, QSlider, QSizePolicy, QFrame,
    QTextEdit, QLineEdit, QApplication
)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import Qt

from predictor import Predictor
from camera import CameraManager
from utils import display_pixmap_scaled, parse_label_file, derive_label_path
from mappings import ENGLISH_TO_CHINESE

class EmotionPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸表情识别系统")
        self.setGeometry(100, 100, 1100, 850)

        self._set_background()

        self.predictor = Predictor(parent_widget=self)

        self.current_image_path = None
        self.confidence_threshold = 0.5

        self.camera = None

        self._setup_ui()

    def _set_background(self):
        self.setStyleSheet("background-color: #f0f4f8;")

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        top_widget = QFrame()
        top_widget.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
            }
        """)
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(25, 25, 25, 25)
        top_layout.setSpacing(25)

        image_group = QFrame()
        image_group.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 15px;
                border: none;
            }
        """)
        image_group_layout = QVBoxLayout(image_group)
        image_group_layout.setContentsMargins(15, 15, 15, 15)

        image_title = QLabel("视图预览")
        image_title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        image_title.setStyleSheet("color: #2c3e50; border: none; background: transparent;")
        image_group_layout.addWidget(image_title)

        self.image_label = QLabel("请选择图片或打开摄像头")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(620, 420)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #dee2e6;
                border-radius: 12px;
                background-color: white;
                color: #6c757d;
                font-size: 14px;
            }
        """)
        image_group_layout.addWidget(self.image_label)

        control_group = QFrame()
        control_group.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 15px;
                border: none;
            }
        """)
        control_group_layout = QVBoxLayout(control_group)
        control_group_layout.setContentsMargins(20, 20, 20, 20)
        control_group_layout.setSpacing(18)

        control_title = QLabel("控制面板")
        control_title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        control_title.setStyleSheet("color: #2c3e50; border: none; background: transparent;")
        control_group_layout.addWidget(control_title)

        self.select_button = QPushButton("选择图片")
        self.select_button.setFont(QFont("Microsoft YaHei", 13))
        self.select_button.setMinimumHeight(50)
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)

        self.camera_button = QPushButton("打开摄像头")
        self.camera_button.setFont(QFont("Microsoft YaHei", 13))
        self.camera_button.setMinimumHeight(50)
        self.camera_button.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """)

        self.predict_button = QPushButton("开始预测")
        self.predict_button.setFont(QFont("Microsoft YaHei", 13))
        self.predict_button.setMinimumHeight(50)
        self.predict_button.setEnabled(False)
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)

        self.model_button = QPushButton("选择模型")
        self.model_button.setFont(QFont("Microsoft YaHei", 13))
        self.model_button.setMinimumHeight(50)
        self.model_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:pressed {
                background-color: #7d3c98;
            }
        """)

        conf_container = QFrame()
        conf_container.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        conf_layout = QVBoxLayout(conf_container)
        conf_layout.setContentsMargins(15, 15, 15, 15)

        conf_label = QLabel("置信度阈值: 0.5")
        conf_label.setFont(QFont("Microsoft YaHei", 12))
        conf_label.setStyleSheet("color: #34495e;")

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.setSingleStep(5)
        self.conf_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                width: 20px;
                height: 20px;
                border-radius: 10px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #2980b9;
            }
        """)

        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)

        control_group_layout.addWidget(self.model_button)
        control_group_layout.addWidget(self.select_button)
        control_group_layout.addWidget(self.camera_button)
        control_group_layout.addWidget(self.predict_button)
        control_group_layout.addWidget(conf_container)
        control_group_layout.addStretch()

        top_layout.addWidget(image_group, stretch=2)
        top_layout.addWidget(control_group, stretch=1)

        result_group = QFrame()
        result_group.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
            }
        """)
        result_group_layout = QVBoxLayout(result_group)
        result_group_layout.setContentsMargins(25, 25, 25, 25)

        result_title = QLabel("结果预览")
        result_title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        result_title.setStyleSheet("color: #2c3e50;")
        result_group_layout.addWidget(result_title)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(200)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                padding: 15px;
                color: #2c3e50;
                font-size: 14px;
                line-height: 1.8;
            }
            QScrollBar:vertical {
                background-color: #f1f2f6;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #bdc3c7;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #95a5a6;
            }
        """)
        result_group_layout.addWidget(self.result_text)

        # Kimi 大模型问答区域
        kimi_group = QFrame()
        kimi_group.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 15px;
                border: none;
            }
        """)
        kimi_group_layout = QVBoxLayout(kimi_group)
        kimi_group_layout.setContentsMargins(15, 15, 15, 15)

        kimi_title = QLabel("Kimi 大模型问答")
        kimi_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        kimi_title.setStyleSheet("color: #2c3e50;")
        kimi_group_layout.addWidget(kimi_title)

        # 问答输入和发送按钮
        chat_container = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("输入问题...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
            }
        """)
        self.send_button = QPushButton("发送")
        self.send_button.setFont(QFont("Microsoft YaHei", 10))
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                margin-left: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        chat_container.addWidget(self.chat_input)
        chat_container.addWidget(self.send_button)
        kimi_group_layout.addLayout(chat_container)

        # 为输入框添加回车键发送功能
        self.chat_input.returnPressed.connect(self.send_kimi_request)

        result_group_layout.addWidget(kimi_group)

        main_layout.addWidget(top_widget)
        main_layout.addWidget(result_group, stretch=1)

        self.model_button.clicked.connect(self.select_model)
        self.select_button.clicked.connect(self.select_image)
        self.predict_button.clicked.connect(self.predict_button_clicked)
        self.camera_button.clicked.connect(self.toggle_camera)
        self.conf_slider.valueChanged.connect(self.on_confidence_changed)
        self.send_button.clicked.connect(self.send_kimi_request)

        self.conf_label = conf_label

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "", "PyTorch模型文件 (*.pt)"
        )
        if file_path:
            if self.predictor.load_new_model(file_path):
                QMessageBox.information(self, "成功", f"模型加载成功！\n{file_path}")
                self.result_text.setHtml(f"已加载新模型：{file_path}")
                if self.camera and self.camera.running:
                    self.camera.predictor = self.predictor

    def on_confidence_changed(self, value):
        self.confidence_threshold = value / 100.0
        self.conf_label.setText(f"置信度阈值: {self.confidence_threshold:.2f}")
        if self.camera and self.camera.predict_enabled:
            self.camera.set_conf(self.confidence_threshold)

    def select_image(self):
        if self.camera and self.camera.running:
            if self.camera.predict_enabled:
                self.camera.stop_detection()
                self.predict_button.setText("开始预测")
            self.camera.stop()
            self.camera_button.setText("打开摄像头")

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            display_pixmap_scaled(self.image_label, pixmap)
            self.predict_button.setEnabled(True)
            self.result_text.setHtml("已选择图片，点击“开始预测”进行推理。")

    def toggle_camera(self):
        if self.camera is None or not self.camera.running:
            if self.camera is None:
                self.camera = CameraManager(self.image_label, predictor=self.predictor)

            if not self.camera.running:
                ok = self.camera.start()
                if not ok:
                    QMessageBox.critical(self, "错误", "无法打开摄像头。")
                    return

            self.camera_button.setText("关闭摄像头")
            self.predict_button.setEnabled(True)
            self.result_text.setHtml("摄像头已打开。点击“开始预测”开启实时检测。")
        else:
            if self.camera.predict_enabled:
                self.camera.stop_detection()
                self.predict_button.setText("开始预测")
            self.camera.stop()
            self.camera_button.setText("打开摄像头")
            self.predict_button.setEnabled(bool(self.current_image_path))
            self.image_label.setText("请选择图片或打开摄像头")
            self.result_text.setHtml("")

    def predict_button_clicked(self):
        if self.camera and self.camera.running:
            if not self.camera.predict_enabled:
                self.camera.start_detection(conf=self.confidence_threshold)
                self.predict_button.setText("停止预测")
                self.result_text.setHtml("正在实时检测摄像头中的人脸表情…")
            else:
                self.camera.stop_detection()
                self.predict_button.setText("开始预测")
                self.result_text.setHtml("摄像头已打开。点击“开始预测”重新开启实时检测。")
            return

        self.predict_image()

    def predict_image(self):
        if not self.current_image_path:
            return
        try:
            image_bgr = cv2.imread(self.current_image_path)
            if image_bgr is None:
                raise ValueError(f"无法读取图片: {self.current_image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            results = self.predictor.predict(image_rgb, conf=self.confidence_threshold)

            label_path = derive_label_path(self.current_image_path)
            true_labels = parse_label_file(label_path)

            result_info = f"<div style='font-size: 15px; line-height: 2;'>"
            result_info += f"<b>图片路径:</b> {self.current_image_path}<br>"
            result_info += f"<b>当前置信度阈值:</b> {self.confidence_threshold}<br><br>"
            has_predictions = False

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                xyxys = boxes.xyxy.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for i in range(len(xyxys)):
                    pred_class_id = int(clss[i])
                    confidence = float(confs[i])
                    en, zh = self.predictor.id_to_names(pred_class_id)

                    is_correct = pred_class_id in true_labels if true_labels else "未知"
                    correct_label = "正确" if is_correct else "错误" if is_correct is not None else "无真实标签"

                    result_info += f"<b>预测类别:</b> {zh} ({en})<br>"
                    result_info += f"<b>置信度:</b> {confidence:.2f}<br>"

                    if true_labels:
                        true_chinese_names = [
                            ENGLISH_TO_CHINESE.get(self.predictor.names[label], self.predictor.names[label])
                            for label in true_labels
                        ]
                        result_info += f"<b>真实类别:</b> {', '.join(true_chinese_names)}<br>"
                        result_info += f"<b>是否准确:</b> {correct_label}<br>"
                    else:
                        result_info += f"<b>真实类别:</b> 无<br>"

                    result_info += "<br><hr style='border: 1px solid #dfe6e9; margin: 10px 0;'><br>"
                    has_predictions = True

            if not has_predictions:
                result_info += f"<b>未检测到任何人脸表情（置信度阈值: {self.confidence_threshold}）</b><br>"

            result_info += "</div>"
            self.result_text.setHtml(result_info)

        except Exception as e:
            QMessageBox.critical(self, "预测错误", f"预测过程中发生错误:<br>{str(e)}")

    def send_kimi_request(self):
        # 硬编码 API Key
        api_key = "sk-hQZTsGNS2XtGgwFOmM0hdWKz5YKPzeDExZ536FgoV5bMCmz0"  # 替换为实际的 API Key
        question = self.chat_input.text().strip()

        if not question:
            QMessageBox.warning(self, "警告", "请输入问题")
            return

        try:
            # 显示加载提示
            self.send_button.setEnabled(False)
            self.send_button.setText("思考中...")
            QApplication.processEvents()  # 刷新界面

            # 构建请求数据
            payload = {
                "model": "moonshot-v1-8k",  # 使用正确的模型名称
                "messages": [
                    {
                        "role": "user",
                        "content": "你的名字叫暮黎助手，你被接入了人脸表情识别系统，主要为用户提供心理咨询服务，请给用户提供专业的心理咨询和情感答疑服务！"
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 4096,
                "top_p": 1,
            }

            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            # 发送请求到 Kimi API
            response = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                json=payload,
                headers=headers
            )

            # 解析响应
            response_data = response.json()
            if "error" in response_data:
                raise Exception(f"API 错误: {response_data['error']['message']}")

            # 获取回答
            answer = response_data["choices"][0]["message"]["content"]

            # 将 Markdown 转换为 HTML
            answer_html = markdown.markdown(answer)

            # 在结果文本框中显示问答
            current_text = self.result_text.toHtml()
            new_text = f"{current_text}<br><hr style='border: 1px solid #dfe6e9; margin: 10px 0;'><br><b>用户:</b> {question}<br><b>Kimi:</b> {answer_html}"
            self.result_text.setHtml(new_text)

            # 自动滚动到文本框底部
            self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())

            # 清空输入框
            self.chat_input.clear()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"调用 Kimi API 时发生错误:<br>{str(e)}")
        finally:
            # 恢复按钮状态
            self.send_button.setEnabled(True)
            self.send_button.setText("发送")

    def closeEvent(self, event):
        if self.camera and self.camera.running:
            self.camera.stop()
        event.accept()
