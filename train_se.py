import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "8"

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f
from pathlib import Path

from attention_modules import SE


class C2f_SE(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.se = SE(c2)
    
    def forward(self, x):
        x = super().forward(x)
        x = self.se(x)
        return x


def modify_model(model, target_layers=[15, 18, 21]):
    print("\n开始修改模型，添加 SE 注意力机制...")
    
    for idx in target_layers:
        if idx < len(model.model.model):
            layer = model.model.model[idx]
            
            if isinstance(layer, C2f):
                c2 = layer.cv2.conv.in_channels
                n = len(layer.m)
                shortcut = layer.m[0].cv1 is not None
                g = 1
                e = 0.5
                
                new_layer = C2f_SE(c2, c2, n, shortcut, g, e)
                
                try:
                    state_dict = layer.state_dict()
                    filtered_dict = {k: v for k, v in state_dict.items() 
                                   if k in new_layer.state_dict().keys() 
                                   and v.shape == new_layer.state_dict()[k].shape}
                    new_layer.load_state_dict(filtered_dict, strict=False)
                    model.model.model[idx] = new_layer
                    print(f"  层 {idx}: C2f (ch={c2}) -> C2f_SE")
                except Exception as e:
                    print(f"  层 {idx}: 权重复制失败 - {e}")
            else:
                print(f"  层 {idx}: 跳过 ({layer.__class__.__name__})")
    
    print("模型修改完成！\n")
    return model


def main():
    data_path = str(Path('YOLO_formatenhance/data.yaml').resolve())
    
    config = {
        'model': 'yolov8m.pt',
        'device': '0',  # GPU 用'0'，CPU 用'cpu'
        'workers': 8,
        'imgsz': 224,
        'epochs': 100,
        'patience': 40,
        'batch': 32,
        
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        
        'cls': 0.5,
        'box': 7.5,
        'dfl': 1.5,
        
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.3,
        'fliplr': 0.5,
        'flipud': 0.1,
        'scale': 0.9,
        'translate': 0.1,
        'degrees': 20.0,
        'shear': 5.0,
        'perspective': 0.001,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'erasing': 0.2,
        
        'dropout': 0.15,
        'amp': True,
        'cos_lr': True,
        
        'project': 'runs/detect',
        'name': 'face_emotion_SE_m_v1'
    }
    
    print("=" * 70)
    print("YOLOv8m - SE Attention Enhanced")
    print("=" * 70)
    print(f"Model: {config['model']}")
    print(f"Attention: SE (Squeeze-and-Excitation)")
    print(f"Target Layers: [15, 18, 21]")
    print(f"Batch: {config['batch']}, LR: {config['lr0']} -> {config['lr0'] * config['lrf']}")
    print(f"Epochs: {config['epochs']} (Early Stop: {config['patience']})")
    print("=" * 70)
    
    print(f"\nLoading model...")
    model = YOLO(config['model'])
    
    model = modify_model(model, target_layers=[15, 18, 21])
    
    print(f"Start training...")
    print("=" * 70)
    
    results = model.train(
        data=data_path,
        device=config['device'],
        workers=config['workers'],
        imgsz=config['imgsz'],
        epochs=config['epochs'],
        patience=config['patience'],
        batch=config['batch'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs'],
        warmup_momentum=config['warmup_momentum'],
        cls=config['cls'],
        box=config['box'],
        dfl=config['dfl'],
        mosaic=config['mosaic'],
        mixup=config['mixup'],
        copy_paste=config['copy_paste'],
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
        erasing=config['erasing'],
        dropout=config['dropout'],
        amp=config['amp'],
        cos_lr=config['cos_lr'],
        project=config['project'],
        name=config['name'],
        verbose=True,
        save=True,
        save_period=-1,
        plots=True
    )
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best model: runs/detect/{config['name']}/weights/best.pt")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    main()
