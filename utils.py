import os
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

def display_pixmap_scaled(label, pixmap: QPixmap):
    """将 QPixmap 等比缩放显示到指定 QLabel（以控件高度为基准）"""
    if pixmap.isNull():
        label.setText("无法加载图片")
        return
    orig_w = pixmap.width()
    orig_h = pixmap.height()
    if orig_h <= 0:
        label.setText("图片高度无效")
        return
    target_height = label.height()  # 与 GUI 中的固定高度一致
    scale_factor = target_height / orig_h
    new_width = int(orig_w * scale_factor)
    new_height = target_height
    scaled_pixmap = pixmap.scaled(
        new_width, new_height,
        Qt.IgnoreAspectRatio,  # 比例已手动保证
        Qt.SmoothTransformation
    )
    label.setPixmap(scaled_pixmap)
    label.setAlignment(Qt.AlignCenter)

def parse_label_file(label_file):
    """解析 YOLO 标签文件，返回 class_id 列表"""
    if not os.path.exists(label_file):
        return []
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            try:
                class_id = int(parts[0])
                labels.append(class_id)
            except ValueError:
                continue
    return labels

def derive_label_path(image_path: str) -> str:
    """由图片路径推导对应的 label 路径（images -> labels，扩展名改为 .txt）"""
    label_path = (
        image_path
        .replace('images', 'labels')
        .replace('.png', '.txt')
        .replace('.jpg', '.txt')
        .replace('.jpeg', '.txt')
    )
    return label_path