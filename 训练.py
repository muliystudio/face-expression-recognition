import os
# 告诉 Intel OpenMP 允许重复加载库（临时绕过报错）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 可选：限制 OpenMP 线程数，避免资源争抢（根据你 CPU 的核心数调整，通常设为物理核心数或4-8）
os.environ["OMP_NUM_THREADS"] = "8"

# 接下来再导入其他库
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def get_abs_path(relative_path):
    return str(Path(relative_path).resolve())


def main():
    # 确保 data.yaml 已经指向了【增强后】的数据集路径
    data_path = get_abs_path('YOLO_formatenhance/data.yaml')

    # 建议改用 's' (small) 模型。
    # 原因：你的数据即使增强了，信息量依然有限。用 'm' 模型参数量太大，极易过拟合。
    # 's' 模型在这个任务上通常足够了，且泛化能力更好。
    model = YOLO('yolov8s.pt', task='detect')

    results = model.train(
        data=data_path,
        device='cuda',
        workers=8,

        # === 输入尺寸 ===
        # 既然我们已经预处理放大了图片，这里保持 224 即可
        imgsz=224,

        # === 训练轮数与早停 ===
        # 既然之前 30 轮就过拟合，这次我们设置少一点，或者依靠 patience
        epochs=100,
        patience=20,  # 如果 20 轮没提升就停，别硬撑

        batch=64,  # s 模型比较小，batch 可以开大点，让梯度更稳

        # === 优化器与学习率 ===
        optimizer='AdamW',
        lr0=0.001,  # s 模型可以用标准学习率

        # === 核心：正则化策略 (防止过拟合) ===
        weight_decay=0.001,  # 【增加】权重衰减，防止模型学得太死
        dropout=0.2,  # 【增加】增加随机失活

        # === 损失权重 ===
        # 之前设得太激进 (cls=4.0)，这次稍微温和点
        cls=2.0,
        box=5.0,

        # === 数据增强 ===
        # 既然图片是“插值”变大的，Mosaic 可能会产生奇怪的伪影，还是建议关闭或开很小
        mosaic=0.1,  # 稍微给一点点，增加背景的多样性
        mixup=0.0,  # 关掉

        # 几何增强（保留）
        fliplr=0.5,
        scale=0.3,  # 缩放不要太剧烈，因为像素本来就捉襟见肘
        degrees=10.0,  # 稍微旋转一下头部

        # === 颜色增强 ===
        # 对抗光照变化，这对表情识别很重要
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,

        project='runs/detect',
        name='face_emotion_enhanced_s_v1-v8n'
    )


if __name__ == '__main__':
    main()