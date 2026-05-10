import os
import urllib.request
import cv2


def download_dnn_model():
    """
    自动下载OpenCV DNN人脸检测模型文件
    """
    # 定义模型文件名和下载URL
    model_file = "opencv_face_detector_uint8.pb"
    config_file = "opencv_face_detector.pbtxt"

    # GitHub上的模型文件URL（官方OpenCV仓库）
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20200519_fp16/res10_300x300_ssd_iter_140000_fp16.pb"
    config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"

    # 获取OpenCV数据目录
    opencv_data_dir = os.path.dirname(cv2.__file__)
    dnn_dir = os.path.join(opencv_data_dir, "data", "dnn")

    # 如果目录不存在则创建
    os.makedirs(dnn_dir, exist_ok=True)

    model_path = os.path.join(dnn_dir, model_file)
    config_path = os.path.join(dnn_dir, config_file)

    # 检查模型文件是否存在
    if os.path.exists(model_path) and os.path.exists(config_path):
        print("DNN模型文件已存在，无需下载。")
        return model_path, config_path

    print("正在下载DNN模型文件...")

    try:
        # 下载模型文件
        print(f"正在下载 {model_file}...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"已下载到: {model_path}")

        # 下载配置文件
        print(f"正在下载 {config_file}...")
        urllib.request.urlretrieve(config_url, config_path)
        print(f"已下载到: {config_path}")

        print("DNN模型下载完成！")
        return model_path, config_path

    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载模型文件并放置到正确路径。")
        return None, None


# 使用示例
if __name__ == "__main__":
    model_path, config_path = download_dnn_model()
    if model_path and config_path:
        print(f"模型路径: {model_path}")
        print(f"配置路径: {config_path}")
    else:
        print("模型下载失败，将使用Haar Cascades作为备选方案。")
