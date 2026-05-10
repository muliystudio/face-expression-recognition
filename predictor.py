import os
from ultralytics import YOLO
from PySide6.QtWidgets import QMessageBox
from mappings import ENGLISH_TO_CHINESE
from config import MODEL_PATH

class Predictor:
    """封装 YOLO 模型的加载与预测"""
    def __init__(self, parent_widget=None):
        self.parent_widget = parent_widget
        self.model_path = MODEL_PATH
        self.model = None
        # 尝试加载模型，但即使失败也不中断程序
        try:
            self._load_model()
        except Exception as e:
            if self.parent_widget is not None:
                QMessageBox.warning(self.parent_widget, "警告", f"默认模型加载失败: {str(e)}\n请通过 '选择模型' 按钮手动选择模型文件。")
    
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        self.model = YOLO(self.model_path)
    
    def load_new_model(self, new_model_path):
        if not os.path.exists(new_model_path):
            return False
        try:
            self.model_path = new_model_path
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            if self.parent_widget is not None:
                QMessageBox.critical(self.parent_widget, "错误", f"加载模型失败: {str(e)}")
            return False

    @property
    def names(self):
        if self.model is None:
            return {}
        return self.model.names

    def predict(self, image_rgb, conf: float):
        """执行预测，返回 ultralytics 的结果对象列表"""
        if self.model is None:
            if self.parent_widget is not None:
                QMessageBox.warning(self.parent_widget, "警告", "模型未加载，请先选择模型文件。")
            return []
        return self.model.predict(image_rgb, conf=conf)

    def id_to_names(self, class_id: int):
        """返回英文名与中文名"""
        if self.model is None:
            return "未知", "未知"
        en = self.names.get(class_id, "未知")
        zh = ENGLISH_TO_CHINESE.get(en, en)
        return en, zh