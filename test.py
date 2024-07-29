# 导入必要的库
import cv2
import torch
from ultralytics import YOLO

# 初始化 YOLOv8 模型
model = YOLO('yolov8n.pt')  # 确保 yolov8n.pt 模型文件在当前目录或正确路径下

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 设置窗口名称
window_name = 'YOLOv8 Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# 设置窗口大小
window_width = 800
window_height = 600
cv2.resizeWindow(window_name, window_width, window_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换图像格式为模型输入格式
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0

    # 进行预测
    with torch.no_grad():
        results = model(img)

    # 绘制预测结果
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{model.names[int(cls)]} {score:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('YOLOv8 Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
