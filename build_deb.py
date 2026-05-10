#!/usr/bin/env python3
"""
DEB打包脚本 - 将人脸表情识别系统打包成Debian/Ubuntu安装包
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# 项目配置
APP_NAME = "face-emotion-recognition"
APP_DISPLAY_NAME = "人脸表情识别系统"
VERSION = "1.0.0"
MAINTAINER = "Your Name <your.email@example.com>"
DESCRIPTION = "基于YOLO的人脸表情识别系统"

def create_deb_structure():
    """创建DEB包目录结构"""
    print("创建DEB包目录结构...")
    
    # 基础目录
    base_dir = Path(f"{APP_NAME}_{VERSION}")
    debian_dir = base_dir / "DEBIAN"
    
    # 安装目录
    opt_dir = base_dir / "opt" / APP_NAME
    bin_dir = base_dir / "usr" / "bin"
    share_dir = base_dir / "usr" / "share"
    applications_dir = share_dir / "applications"
    icons_dir = share_dir / "icons" / "hicolor" / "256x256" / "apps"
    
    # 创建目录
    for dir_path in [debian_dir, opt_dir, bin_dir, applications_dir, icons_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'base': base_dir,
        'debian': debian_dir,
        'opt': opt_dir,
        'bin': bin_dir,
        'applications': applications_dir,
        'icons': icons_dir
    }

def create_control_file(debian_dir):
    """创建DEB控制文件"""
    print("创建control文件...")
    
    control_content = f"""Package: {APP_NAME}
Version: {VERSION}
Section: science
Priority: optional
Architecture: amd64
Depends: python3 (>= 3.8), python3-pip, python3-venv, libgl1-mesa-glx, libglib2.0-0, libsm6, libxext6, libxrender-dev, libgomp1
Maintainer: {MAINTAINER}
Description: {DESCRIPTION}
 基于YOLO深度学习框架的人脸表情识别系统，
 支持图片识别和实时摄像头检测。
"""
    
    control_path = debian_dir / "control"
    with open(control_path, 'w', encoding='utf-8') as f:
        f.write(control_content)
    
    # 创建postinst脚本（安装后执行）
    postinst_content = f"""#!/bin/bash
set -e

echo "正在安装 {APP_DISPLAY_NAME}..."

# 创建虚拟环境
VENV_DIR="/opt/{APP_NAME}/venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# 激活虚拟环境并安装依赖
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# 安装项目依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python-headless PySide6 numpy matplotlib scipy pandas tqdm requests

echo "{APP_DISPLAY_NAME} 安装完成！"
echo "可以通过命令 'face-emotion' 启动程序"
exit 0
"""
    
    postinst_path = debian_dir / "postinst"
    with open(postinst_path, 'w', encoding='utf-8') as f:
        f.write(postinst_content)
    os.chmod(postinst_path, 0o755)
    
    # 创建prerm脚本（卸载前执行）
    prerm_content = f"""#!/bin/bash
set -e

echo "正在卸载 {APP_DISPLAY_NAME}..."

# 可以在这里添加清理逻辑

exit 0
"""
    
    prerm_path = debian_dir / "prerm"
    with open(prerm_path, 'w', encoding='utf-8') as f:
        f.write(prerm_content)
    os.chmod(prerm_path, 0o755)

def copy_project_files(opt_dir):
    """复制项目文件到安装目录"""
    print("复制项目文件...")
    
    # 需要复制的文件和目录
    files_to_copy = [
        'maingui.py',
        'gui.py',
        'login.py',
        'predictor.py',
        'camera.py',
        'utils.py',
        'mappings.py',
        'config.py',
        'attention_modules.py',
        'assets',
        'runs/detect/face_emotion_enhanced_s_v1-v8m/weights/best.pt'
    ]
    
    for item in files_to_copy:
        src = Path(item)
        dst = opt_dir / item
        
        if src.exists():
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            print(f"  复制: {item}")
        else:
            print(f"  警告: {item} 不存在，跳过")

def create_launcher(bin_dir, opt_dir):
    """创建启动脚本"""
    print("创建启动脚本...")
    
    launcher_content = f"""#!/bin/bash
# {APP_DISPLAY_NAME} 启动脚本

VENV_DIR="/opt/{APP_NAME}/venv"
APP_DIR="/opt/{APP_NAME}"

# 检查虚拟环境是否存在
if [ ! -d "$VENV_DIR" ]; then
    echo "错误：虚拟环境不存在，请重新安装软件包"
    exit 1
fi

# 激活虚拟环境并运行
source "$VENV_DIR/bin/activate"
cd "$APP_DIR"
python3 maingui.py "$@"
"""
    
    launcher_path = bin_dir / "face-emotion"
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(launcher_content)
    os.chmod(launcher_path, 0o755)

def create_desktop_entry(applications_dir, icons_dir):
    """创建桌面快捷方式"""
    print("创建桌面快捷方式...")
    
    # 创建简单的图标
    icon_path = icons_dir / f"{APP_NAME}.png"
    # 如果有图标文件，复制它；否则创建一个简单的文本图标
    if Path("assets/icon.png").exists():
        shutil.copy2("assets/icon.png", icon_path)
    else:
        # 创建一个简单的占位图标
        with open(icon_path, 'w') as f:
            f.write("")  # 空文件作为占位符
    
    desktop_content = f"""[Desktop Entry]
Name={APP_DISPLAY_NAME}
Name[zh_CN]={APP_DISPLAY_NAME}
Comment={DESCRIPTION}
Comment[zh_CN]={DESCRIPTION}
Exec=/usr/bin/face-emotion
Icon={APP_NAME}
Terminal=false
Type=Application
Categories=Science;Education;Graphics;
Keywords=face;emotion;recognition;deep learning;
StartupNotify=true
"""
    
    desktop_path = applications_dir / f"{APP_NAME}.desktop"
    with open(desktop_path, 'w', encoding='utf-8') as f:
        f.write(desktop_content)

def build_deb(base_dir):
    """构建DEB包"""
    print("构建DEB包...")
    
    # 计算安装大小
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(base_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    
    # 更新control文件中的Installed-Size
    control_path = base_dir / "DEBIAN" / "control"
    with open(control_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content += f"\nInstalled-Size: {total_size // 1024}\n"
    with open(control_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 构建DEB包
    deb_name = f"{APP_NAME}_{VERSION}_amd64.deb"
    result = subprocess.run(
        ['dpkg-deb', '--build', str(base_dir), deb_name],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"[OK] DEB包构建成功: {deb_name}")
        return deb_name
    else:
        print(f"[ERROR] 构建失败: {result.stderr}")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print(f"开始构建 {APP_DISPLAY_NAME} DEB包")
    print("=" * 60)
    
    # 检查是否在Linux系统上
    if sys.platform != 'linux':
        print("[!] 警告：当前不是Linux系统，DEB包需要在Linux上构建")
        print("   生成的脚本可以在Linux/WSL上运行")
    
    # 检查dpkg-deb是否可用
    if shutil.which('dpkg-deb'):
        print("[OK] 找到 dpkg-deb 工具")
    else:
        print("[!] 未找到 dpkg-deb，只能生成目录结构，无法构建DEB包")
        print("   在Debian/Ubuntu上安装: sudo apt-get install dpkg-dev")
    
    try:
        # 创建目录结构
        dirs = create_deb_structure()
        
        # 创建控制文件
        create_control_file(dirs['debian'])
        
        # 复制项目文件
        copy_project_files(dirs['opt'])
        
        # 创建启动脚本
        create_launcher(dirs['bin'], dirs['opt'])
        
        # 创建桌面快捷方式
        create_desktop_entry(dirs['applications'], dirs['icons'])
        
        # 构建DEB包
        if shutil.which('dpkg-deb'):
            deb_file = build_deb(dirs['base'])
            if deb_file:
                print(f"\n[PACKAGE] 安装命令:")
                print(f"   sudo dpkg -i {deb_file}")
                print(f"   sudo apt-get install -f  # 修复依赖")
                print(f"\n[RUN] 运行命令:")
                print(f"   face-emotion")
        else:
            print(f"\n[DIR] 目录结构已创建: {dirs['base']}")
            print("   请在Linux系统上运行 dpkg-deb --build 来构建DEB包")
        
        print("\n" + "=" * 60)
        print("构建完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
