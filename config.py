# 模型与默认参数配置
import os
import sys

# 获取应用程序运行目录
if hasattr(sys, '_MEIPASS'):
    # 打包后的环境
    BASE_DIR = sys._MEIPASS
else:
    # 开发环境
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径
MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'face_emotion_enhanced_s_v1-v8m', 'weights', 'best.pt')

# 预览区域默认大小（与 GUI 左侧图片预览栏一致）
PREVIEW_WIDTH = 600
PREVIEW_HEIGHT = 400

# 摄像头帧率（定时器间隔）
CAMERA_INTERVAL_MS = 30  # ~33ms ≈ 30fps