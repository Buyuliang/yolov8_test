import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# 初始化 MTCNN 和 ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)  # keep_all=False 只返回第一个检测到的面部
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face_embedding(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    if face_tensor is not None:
        face_tensor = face_tensor.unsqueeze(0).to(device)
        embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
        return embedding
    else:
        print(f"No face detected in {img_path}")
        return None

def save_embeddings_from_directory(directory_path):
    embeddings = {}
    for person_name in os.listdir(directory_path):
        person_dir = os.path.join(directory_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        person_embeddings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            embedding = extract_face_embedding(img_path)
            if embedding is not None:
                person_embeddings.append(embedding)
        if person_embeddings:
            embeddings[person_name] = np.mean(person_embeddings, axis=0).tolist()
    return embeddings

# 示例目录结构
# known_faces/
# ├── person1/
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   └── ...
# ├── person2/
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   └── ...
# └── ...

directory_path = "info"
known_embeddings = save_embeddings_from_directory(directory_path)

# 保存嵌入向量到文件
import json
with open("known_embeddings.json", "w") as f:
    json.dump(known_embeddings, f)
