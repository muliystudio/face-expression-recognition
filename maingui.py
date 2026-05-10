import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from gui import EmotionPredictorApp
from login import LoginWindow

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    try:
        main_window = EmotionPredictorApp()
        login_window = LoginWindow()
        login_window.set_main_window(main_window)
    except Exception as e:
        print(f"初始化失败: {e}")
        sys.exit(1)
    login_window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()