# 人脸表情识别系统

## 项目简介

本项目是一个基于Qt和YOLO的人脸表情识别系统，能够实时检测和识别人脸表情。

## 功能特点

- 支持图片上传和识别
- 支持摄像头实时检测
- 可调节置信度阈值
- 显示预测结果和真实标签对比

## 如何使用

### 直接运行可执行文件

1. 进入 `dist` 目录
2. 双击 `人脸表情识别系统.exe` 运行应用程序
3. 在应用界面中：
   - 点击"选择图片"按钮上传图片
   - 点击"打开摄像头"按钮启动摄像头
   - 调整置信度阈值
   - 点击"开始预测"按钮进行识别

### 从源码运行

1. 安装依赖：
   ```bash
   pip install PySide6 opencv-python ultralytics
   ```
2. 运行主文件：
   ```bash
   python maingui.py
   ```

## 项目结构

- `maingui.py` - 主程序入口
- `gui.py` - 界面实现
- `predictor.py` - 模型预测封装
- `camera.py` - 摄像头管理
- `utils.py` - 工具函数
- `mappings.py` - 标签映射
- `config.py` - 配置文件
- `runs/` - 模型文件目录

## 打包说明

本项目使用 PyInstaller 进行打包，生成了独立的可执行文件。打包过程包含了所有必要的依赖和模型文件，确保应用程序可以在未安装Qt开发环境的Windows系统上独立运行。

### 打包命令

```bash
python package.py
```

### 打包结果

- `dist/人脸表情识别系统.exe` - 主可执行文件
- 所有必要的依赖和模型文件已嵌入到可执行文件中

## 注意事项

1. 首次运行可能会稍微慢一些，因为需要加载模型
2. 确保摄像头权限已开启
3. 识别结果的准确性取决于模型训练质量

## 技术栈

- Python 3.14+
- PySide6 (Qt for Python)
- OpenCV
- Ultralytics YOLO
- PyInstaller (打包工具)