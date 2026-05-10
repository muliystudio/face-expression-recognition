import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from utils import display_pixmap_scaled
from config import CAMERA_INTERVAL_MS

# 每个类别对应一个颜色（BGR）
CLASS_COLORS = [
    (0, 140, 255),   # 橙 - 愤怒
    (255, 0, 0),     # 蓝 - 轻蔑
    (0, 255, 0),     # 绿 - 厌恶
    (255, 0, 255),   # 洋红 - 恐惧
    (0, 255, 255),   # 黄 - 开心
    (255, 0, 127),   # 紫 - 中性
    (0, 165, 255),   # 橙红 - 悲伤
    (255, 255, 0),   # 青 - 惊讶
]

class CameraManager:
    """管理摄像头的开启、关闭与帧更新显示；使用OpenCV DNN检测人脸，YOLO模型识别表情"""
    def __init__(self, image_label, predictor=None, use_directshow=True):
        self.image_label = image_label
        self.cap = None
        self.timer = None
        self.running = False

        # 检测相关
        self.predictor = predictor
        self.predict_enabled = False
        self.conf_threshold = 0.5

        self.use_directshow = use_directshow  # Windows 上 CAP_DSHOW 更稳

        # 初始化OpenCV DNN人脸检测器
        self.face_net = None
        self.face_cascade = None
        self.use_opencv_dnn = False
        self._init_face_detectors()

    def _init_face_detectors(self):
        """初始化人脸检测器，只使用Haar Cascades"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade is not None and not self.face_cascade.empty():
                print("使用Haar Cascades人脸检测器")
            else:
                print("Haar Cascades加载失败或为空")
                self.face_cascade = None
        except Exception as e:
            print(f"加载Haar Cascades时出错: {e}")
            self.face_cascade = None
        self.use_opencv_dnn = False

    def start(self) -> bool:
        if self.running:
            return True
        api = cv2.CAP_DSHOW if self.use_directshow else 0
        self.cap = cv2.VideoCapture(0, api)
        if not self.cap or not self.cap.isOpened():
            self.cap = None
            return False

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(CAMERA_INTERVAL_MS)

        self.running = True
        return True

    def stop(self):
        if self.timer:
            self.timer.stop()
            self.timer = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.running = False
        self.predict_enabled = False

    def start_detection(self, conf: float):
        """开始实时检测（需要 predictor 已注入）"""
        if self.predictor is None:
            return
        self.predict_enabled = True
        self.conf_threshold = conf

    def stop_detection(self):
        """停止实时检测，仅预览画面"""
        self.predict_enabled = False

    def set_conf(self, conf: float):
        self.conf_threshold = conf

    def _detect_faces_opencv_dnn(self, frame):
        """使用OpenCV DNN检测人脸"""
        if self.face_net is None:
            return []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # 人脸检测置信度阈值
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2, y2))
        return faces

    def _detect_faces_haar(self, frame):
        """使用Haar Cascades检测人脸"""
        if self.face_cascade is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        face_rects = []
        for (x, y, w, h) in faces:
            # 调整框的大小，使其更贴合人脸
            # 扩大一点以包含完整面部
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.15)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)
            face_rects.append((x1, y1, x2, y2))
        return face_rects

    def _detect_faces(self, frame):
        """统一的人脸检测接口"""
        if self.face_cascade is not None:
            return self._detect_faces_haar(frame)
        else:
            return []

    def _draw_box_and_label(self, frame_bgr, x1, y1, x2, y2, cls_id, conf, label_text):
        """绘制精确的人脸框和标签"""
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        # 画精确的人脸框
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        # 在人脸框的右上角显示预测结果
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        # 使用英文标签而不是中文，避免显示问号
        en, zh = self.predictor.id_to_names(cls_id)
        result_text = f"{en} {conf:.2f}"

        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(result_text, font, scale, thickness)

        # 在右上角显示结果
        text_x = x2 - text_width - 6
        text_y = y1 + text_height + 6

        # 确保文本不会超出图像边界
        text_x = max(text_x, 0)
        text_y = min(text_y, frame_bgr.shape[0])

        # 绘制半透明背景
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (text_x - 3, text_y - text_height - 3),
                      (text_x + text_width + 3, text_y + baseline + 3), color, -1)
        cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

        # 绘制白色文字
        cv2.putText(frame_bgr, result_text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _update_frame(self):
        if not self.cap:
            return
        ret, frame_bgr = self.cap.read()
        if not ret:
            return

        if self.predict_enabled and self.predictor is not None:
            # 先使用OpenCV检测精确的人脸位置
            faces = self._detect_faces(frame_bgr)

            # 对每个检测到的人脸进行表情识别
            for (x1, y1, x2, y2) in faces:
                # 裁剪人脸区域
                face_img = frame_bgr[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                # 转换为RGB格式用于YOLO预测
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                # 使用YOLO模型进行表情预测
                results = self.predictor.predict(face_rgb, conf=self.conf_threshold)

                # 处理预测结果
                has_prediction = False
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    clss = boxes.cls.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    if len(clss) > 0:
                        # 取置信度最高的预测结果
                        best_idx = np.argmax(confs)
                        cls_id = int(clss[best_idx])
                        conf = float(confs[best_idx])
                        en, zh = self.predictor.id_to_names(cls_id)
                        label_text = zh

                        # 绘制框和标签
                        self._draw_box_and_label(frame_bgr, x1, y1, x2, y2, cls_id, conf, label_text)
                        has_prediction = True
                        break

                # 如果没有预测结果，只绘制人脸框（可选）
                if not has_prediction:
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # 显示到左侧预览
        frame_rgb2 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb2.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb2.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        display_pixmap_scaled(self.image_label, pixmap)
