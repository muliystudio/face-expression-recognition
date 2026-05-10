import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f
from attention_modules import CBAM


class C2f_CBAM(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.cbam = CBAM(c2)
    
    def forward(self, x):
        x = super().forward(x)
        x = self.cbam(x)
        return x


def modify_model(model, target_layers=[15, 18, 21]):
    print("\nModifying model, adding CBAM attention mechanism...")
    
    for idx in target_layers:
        if idx < len(model.model.model):
            layer = model.model.model[idx]
            
            if isinstance(layer, C2f):
                c2 = layer.cv2.conv.in_channels
                n = len(layer.m)
                shortcut = layer.m[0].cv1 is not None
                g = 1
                e = 0.5
                
                new_layer = C2f_CBAM(c2, c2, n, shortcut, g, e)
                
                try:
                    state_dict = layer.state_dict()
                    filtered_dict = {k: v for k, v in state_dict.items() 
                                   if k in new_layer.state_dict().keys() 
                                   and v.shape == new_layer.state_dict()[k].shape}
                    new_layer.load_state_dict(filtered_dict, strict=False)
                    model.model.model[idx] = new_layer
                    print(f"  Layer {idx}: C2f (ch={c2}) -> C2f_CBAM")
                except Exception as e:
                    print(f"  Layer {idx}: Weight copy failed - {e}")
            else:
                print(f"  Layer {idx}: Skipped ({layer.__class__.__name__})")
    
    print("Model modification completed!\n")
    return model


def print_model_layers(model, title):
    print(f"\n{title}:")
    print("=" * 80)
    for i, layer in enumerate(model.model.model):
        layer_name = layer.__class__.__name__
        if hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
            if hasattr(layer, 'cbam'):
                print(f"{i:3d}: {layer_name} (CBAM) - ch={layer.cv2.conv.in_channels}")
            else:
                print(f"{i:3d}: {layer_name} - ch={layer.cv2.conv.in_channels}")
        else:
            print(f"{i:3d}: {layer_name}")
    print("=" * 80)


def main():
    print("Loading YOLOv8m model...")
    model = YOLO('yolov8m.pt')
    
    # Print original model structure
    print_model_layers(model, "Original Model Structure")
    
    # Modify model
    model = modify_model(model, target_layers=[15, 18, 21])
    
    # Print modified model structure
    print_model_layers(model, "Modified Model Structure")
    
    print("\nModel replacement comparison completed!")
    print("You can see that layers 15, 18, 21 have been replaced from C2f to C2f_CBAM")


if __name__ == '__main__':
    main()
