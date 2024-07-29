# 导入ultralytics的YOLO库
from ultralytics import YOLO

# 加载模型
model = YOLO(r"G:\yolo\ultralytics\runs\detect\train\weights\best.pt")  # 加载自定义的训练模型

if __name__ == '__main__':

    results = model.predict(
        r"C:\Users\Administrator\Desktop\car.jpg",
        save=True, imgsz=640, conf=0.5,iou=0.5)