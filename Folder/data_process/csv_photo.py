import os
import pandas as pd
import numpy as np
from PIL import Image

# 表情类别映射，根据FER2013官方定义
emotion_map = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# 加载数据集
data = pd.read_csv('fer2013.csv')

# 创建根目录
base_dir = 'fer2013'
datasets = ['train', 'val', 'test']

for dataset in datasets:
    for emotion_name in emotion_map.values():
        os.makedirs(os.path.join(base_dir, dataset, emotion_name), exist_ok=True)

# 遍历数据集，提取并保存图像
for index, row in data.iterrows():
    emotion = emotion_map[row['emotion']]
    usage = row['Usage'].lower()
    if usage == 'training':
        dataset_type = 'train'
    elif usage == 'publictest':
        dataset_type = 'val'
    elif usage == 'privatetest':
        dataset_type = 'test'
    else:
        continue

    pixels = np.array(list(map(int, row['pixels'].split()))).reshape(48, 48)
    img = Image.fromarray(pixels.astype(np.uint8))

    filename = os.path.join(base_dir, dataset_type, emotion, f"{dataset_type}_{index}_{emotion}.jpg")
    img.save(filename)
    if index % 1000 == 0:
        print(f"Processed {index} images")

print("所有图像已成功提取并分类到train、val、test的各自目录中。");

