import os
import subprocess
import shutil
import sys

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 清理旧的构建文件
def clean_old_build():
    for dir_name in ['build', 'dist', '__pycache__']:
        dir_path = os.path.join(ROOT_DIR, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"清理目录: {dir_path}")

# 安装必要的依赖
def install_dependencies():
    print("安装必要的依赖...")
    # 使用默认的PyPI源，避免清华镜像源的访问问题
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller", "--index-url", "https://pypi.org/simple/"], check=True)

# 执行打包
def package_app():
    print("开始打包应用...")
    
    # 打包命令，明确指定需要包含的模块
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--name", "人脸表情识别系统",
        "--onefile",
        "--windowed",  # 无控制台窗口
        "--add-data", f"runs/detect/face_emotion_v8n/weights/best.pt;runs/detect/face_emotion_v8m/weights",
        "--add-data", f"assets/background.jpg;assets",
        "--hidden-import", "PySide6",
        "--hidden-import", "PySide6.QtWidgets",
        "--hidden-import", "PySide6.QtGui",
        "--hidden-import", "PySide6.QtCore",
        "--hidden-import", "cv2",
        "--hidden-import", "ultralytics",
        "--hidden-import", "numpy",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "PIL",
        "--hidden-import", "matplotlib",
        "--hidden-import", "scipy",
        "--hidden-import", "tqdm",
        "--hidden-import", "requests",
        "--hidden-import", "pandas",
        "--hidden-import", "markdown",
        "--collect-data", "cv2",
        "--collect-data", "ultralytics",
        "--collect-data", "torch",
        "maingui.py"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# 复制必要的文件到dist目录
def copy_necessary_files():
    dist_dir = os.path.join(ROOT_DIR, "dist")
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    
    # 检查是否需要复制其他文件
    print("打包完成！")

if __name__ == "__main__":
    try:
        # 安装依赖（使用默认PyPI源）
        install_dependencies()
        # 跳过清理步骤，直接打包
        package_app()
        copy_necessary_files()
        print("\n打包成功！可执行文件已生成在 dist 目录中。")
    except Exception as e:
        print(f"打包失败: {e}")
        sys.exit(1)