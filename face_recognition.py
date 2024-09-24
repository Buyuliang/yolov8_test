import os
import cv2
import torch
import numpy as np
import json
from facenet_pytorch import MTCNN, InceptionResnetV1

# 初始化 MTCNN 和 ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)  # keep_all=False 只返回第一个检测到的面部
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 加载已知人脸嵌入向量
with open("known_embeddings.json", "r") as f:
    known_embeddings = json.load(f)

# 提取人脸嵌入向量的函数
def extract_face_embedding(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    if face_tensor is not None:
        face_tensor = face_tensor.unsqueeze(0).to(device)
        embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
        return embedding
    else:
        return None

# 匹配嵌入向量的函数
def compare_embeddings(embedding, known_embeddings, threshold=0.6):
    min_dist = float('inf')
    match = "Unknown"
    for name, known_embedding in known_embeddings.items():
        dist = np.linalg.norm(embedding - np.array(known_embedding))
        if dist < min_dist:
            min_dist = dist
            match = name if dist < threshold else "Unknown"
    return match

# 实时人脸识别
cap = cv2.VideoCapture(0)  # 使用摄像头
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = mtcnn.detect(frame)[0]
    if faces is not None:
        for box in faces:
            x1, y1, x2, y2 = [int(val) for val in box]
            face = frame[y1:y2, x1:x2]
            embedding = extract_face_embedding(face)
            if embedding is not None:
                match = compare_embeddings(embedding, known_embeddings)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
