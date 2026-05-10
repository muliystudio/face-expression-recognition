import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    """CBAM 注意力模块 - 通道 + 空间双重注意力"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 通道注意力
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        channel_att = self.channel_fc(avg_out) + self.channel_fc(max_out)
        channel_att = channel_att.view(b, c, 1, 1)
        channel_att = self.sigmoid(channel_att)
        
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_conv(spatial_input)
        spatial_att = self.sigmoid(spatial_att)
        
        x = x * spatial_att
        
        return x


class CoordinateAttention(nn.Module):
    """坐标注意力 - 保留位置信息的注意力机制"""
    
    def __init__(self, channels, reduction=32):
        super().__init__()
        
        self.directions = [
            (1, 0),  # 水平方向
            (0, 1)   # 垂直方向
        ]
        
        # 水平方向池化
        self.x_avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        # 垂直方向池化
        self.y_avg_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # 融合两个方向的特征
        self.conv = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        
        # 分别生成水平和垂直的注意力权重
        self.conv_x = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.conv_y = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 水平方向全局池化
        x_h = self.x_avg_pool(x)  # (b, c, h, 1)
        # 垂直方向全局池化
        x_w = self.y_avg_pool(x)  # (b, c, 1, w)
        
        # 拼接两个方向的特征
        y = torch.cat([x_h, x_w], dim=2)  # (b, c, h+w, 1)
        y = y.permute(0, 2, 1, 3)  # (b, h+w, c, 1)
        
        # 降维
        y = self.conv(y)
        y = self.bn(y)
        y = self.relu(y)
        
        # 分离水平和垂直特征
        x_h, x_w = torch.split(y, [h, w], dim=1)
        x_h = x_h.permute(0, 2, 1, 3)  # (b, c, h, 1)
        x_w = x_w.permute(0, 2, 1, 3)  # (b, c, w, 1)
        
        # 生成注意力权重
        att_x = self.sigmoid(self.conv_x(x_h))  # (b, c, h, 1)
        att_y = self.sigmoid(self.conv_y(x_w))  # (b, c, 1, w)
        
        # 应用注意力
        out = x * att_x * att_y
        
        return out


class EMA(nn.Module):
    """Efficient Multi-Scale Attention - 高效多尺度注意力"""
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # 分组卷积实现多尺度特征提取
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // reduction)
        
        # 不同大小的深度卷积
        self.conv3x3 = nn.Conv2d(
            channels // reduction, channels // reduction,
            kernel_size=3, padding=1, groups=channels // reduction, bias=False
        )
        self.conv5x5 = nn.Conv2d(
            channels // reduction, channels // reduction,
            kernel_size=5, padding=2, groups=channels // reduction, bias=False
        )
        
        self.bn2 = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        self.conv_out = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        
        # 降维
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 多尺度特征提取
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        
        # 融合多尺度特征
        x = x3 + x5
        x = self.bn2(x)
        x = self.relu(x)
        
        # 通道注意力
        att = self.channel_att(x)
        x = x * att
        
        # 恢复维度
        x = self.conv_out(x)
        x = self.bn_out(x)
        
        # 残差连接
        out = x + identity
        out = F.relu(out)
        
        return out


class SE(nn.Module):
    """SE 注意力 - 经典的通道注意力"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y


def create_attention(attention_type, channels):
    """工厂函数 - 创建指定类型的注意力模块"""
    
    attention_map = {
        'cbam': CBAM,
        'coordinate': CoordinateAttention,
        'ema': EMA,
        'se': SE
    }
    
    if attention_type not in attention_map:
        raise ValueError(f"不支持的注意力类型：{attention_type}")
    
    return attention_map[attention_type](channels)
