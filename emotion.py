import cv2
import numpy as np
from ultralytics import YOLO

# 加载YOLOv8表情分类模型
yolo_model = YOLO('emotion.pt')  # 替换为你的YOLOv8表情分类模型路径

# 定义表情类别
emotions = ['Angry', 'Fearful', 'Happy', 'Neutral', 'Sad']

def main():
    # 打开视频流或读取图像
    video_capture = cv2.VideoCapture(0)  # 0表示打开默认摄像头

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # 使用YOLOv8进行人脸检测和表情分类
        results = yolo_model(frame)

        # 遍历检测到的结果
        for result in results:
            boxes = result.boxes
            # scores = result.scores
            classes = result.classes

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # confidence = scores[i]
                class_id = int(classes[i])

                # 取出人脸区域
                face = frame[y1:y2, x1:x2]
                
                # 在这里直接用YOLO模型进行分类
                # 注意：如果YOLOv8的模型已经进行分类，这一步可以省略
                # emotion = emotions[class_id]

                # 在图像上绘制人脸框和预测的表情
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Face Emotion Detection', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
