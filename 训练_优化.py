import os
import time
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import yaml

# 环境变量设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "8"


def get_abs_path(relative_path):
    """获取绝对路径"""
    return str(Path(relative_path).resolve())


def load_config():
    """加载配置文件"""
    config = {
        # 数据配置
        'data_path': get_abs_path('YOLO_formatenhance/data.yaml'),
        'model_type': 'yolov8m',  # 推荐使用s模型
        
        # 训练配置
        'epochs': 100,
        'patience': 25,  # 早停轮数
        'batch_size': 64,
        'workers': 8,
        'imgsz': 224,
        
        # 优化器配置
        'optimizer': 'AdamW',
        'lr0': 0.001,  # 初始学习率
        'lrf': 0.01,   # 最终学习率 (lr0 * lrf)
        'weight_decay': 0.001,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        
        # 正则化
        'dropout': 0.2,
        
        # 损失权重
        'cls': 2.5,  # 分类损失权重
        'box': 5.0,  # 边界框损失权重
        'dfl': 1.5,  # 分布 focal loss 权重
        
        # 数据增强
        'mosaic': 0.2,  # 马赛克增强
        'mixup': 0.0,   # 混合增强
        'fliplr': 0.5,  # 水平翻转
        'flipud': 0.0,  # 垂直翻转（不推荐）
        'scale': 0.2,   # 缩放范围
        'translate': 0.1,  # 平移
        'degrees': 15.0,  # 旋转角度
        'shear': 5.0,    # 剪切
        'perspective': 0.001,  # 透视变换
        
        # 颜色增强（对表情识别很重要）
        'hsv_h': 0.015,  # 色调
        'hsv_s': 0.4,    # 饱和度
        'hsv_v': 0.4,    # 亮度
        
        # 输出配置
        'project': 'runs/detect',
        'name': f'face_emotion_{time.strftime("%Y%m%d_%H%M%S")}',
        'save_period': 10,  # 每10轮保存一次
        'plots': True,  # 生成训练曲线
        'exist_ok': True  # 允许覆盖现有目录
    }
    return config


def validate_data_config(data_path):
    """验证数据配置文件"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in data_config:
                raise ValueError(f"数据配置文件缺少必要键: {key}")
        
        # 检查路径是否存在
        data_dir = Path(data_path).parent  # data.yaml 所在目录
        for split in ['train', 'val']:
            split_path = data_dir / data_config[split]
            if not split_path.exists():
                raise ValueError(f"数据集路径不存在: {split_path}")
            print(f"  {split} 路径: {split_path}")
        
        print(f"✓ 数据配置验证成功")
        print(f"  类别数: {data_config['nc']}")
        print(f"  类别: {data_config['names']}")
        return True
        
    except Exception as e:
        print(f"✗ 数据配置验证失败: {e}")
        return False


def setup_model(model_type):
    """设置模型"""
    print(f"加载 {model_type} 模型...")
    model = YOLO(f'{model_type}.pt', task='detect')
    print(f"✓ 模型加载成功")
    return model


def train_model(model, config):
    """训练模型"""
    print("=" * 80)
    print("开始训练")
    print("=" * 80)
    
    # 打印配置信息
    print(f"模型: {config['model_type']}")
    print(f"数据集: {config['data_path']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"训练轮数: {config['epochs']}")
    print(f"输入尺寸: {config['imgsz']}")
    print(f"优化器: {config['optimizer']}")
    print(f"初始学习率: {config['lr0']}")
    print(f"早停耐心: {config['patience']}")
    print("=" * 80)
    
    # 开始训练
    start_time = time.time()
    results = model.train(
        data=config['data_path'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        workers=config['workers'],
        imgsz=config['imgsz'],
        epochs=config['epochs'],
        patience=config['patience'],
        batch=config['batch_size'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs'],
        warmup_momentum=config['warmup_momentum'],
        dropout=config['dropout'],
        cls=config['cls'],
        box=config['box'],
        dfl=config['dfl'],
        mosaic=config['mosaic'],
        mixup=config['mixup'],
        fliplr=config['fliplr'],
        flipud=config['flipud'],
        scale=config['scale'],
        translate=config['translate'],
        degrees=config['degrees'],
        shear=config['shear'],
        perspective=config['perspective'],
        hsv_h=config['hsv_h'],
        hsv_s=config['hsv_s'],
        hsv_v=config['hsv_v'],
        project=config['project'],
        name=config['name'],
        save_period=config['save_period'],
        plots=config['plots'],
        exist_ok=config['exist_ok'],
        verbose=True
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print("=" * 80)
    print("训练完成")
    print(f"训练时间: {training_time:.2f} 秒 ({training_time/3600:.2f} 小时)")
    print(f"最佳模型: {results.save_dir}/weights/best.pt")
    print("=" * 80)
    
    return results


def evaluate_model(model, config):
    """评估模型"""
    print("开始评估模型...")
    
    # 加载最佳权重
    best_model_path = Path(config['project']) / config['name'] / 'weights' / 'best.pt'
    if best_model_path.exists():
        eval_model = YOLO(str(best_model_path))
        print(f"加载最佳模型: {best_model_path}")
        
        # 评估验证集
        eval_results = eval_model.val(
            data=config['data_path'],
            imgsz=config['imgsz'],
            batch=config['batch_size'],
            workers=config['workers'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("评估结果:")
        print(f"mAP@0.5: {eval_results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {eval_results.box.map:.4f}")
        print(f"Precision: {eval_results.box.mp:.4f}")
        print(f"Recall: {eval_results.box.mr:.4f}")
        
        return eval_results
    else:
        print(f"✗ 最佳模型文件不存在: {best_model_path}")
        return None


def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    # 验证数据配置
    if not validate_data_config(config['data_path']):
        print("数据配置验证失败，退出训练")
        return
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"✓ GPU 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU 不可用，使用 CPU 训练")
    
    # 设置模型
    model = setup_model(config['model_type'])
    
    # 训练模型
    results = train_model(model, config)
    
    # 评估模型
    evaluate_model(model, config)


if __name__ == '__main__':
    main()
