import numpy as np
from collections import defaultdict

# 原始RAF-DB 7类数据量
class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
class_counts = np.array([1441, 545, 796, 5965, 2475, 1615, 2502])

# 平衡每类样本量（取最小值545）
# balanced_count = min(class_counts)
balanced_count = 1000
print(f"平衡后每类样本数: {balanced_count}")

# 生成平衡后的数据索引（使用元组 (class_name, index) 避免冲突）
balanced_indices = {}
for cls, count in zip(class_names, class_counts):
    indices = list(range(count))
    if count > balanced_count:
        balanced_indices[cls] = [(cls, idx) for idx in np.random.choice(indices, balanced_count, replace=False)]
    else:
        balanced_indices[cls] = [(cls, idx) for idx in np.random.choice(indices, balanced_count, replace=True)]

# 参数设置
num_clients = 10
alpha = 100000  # 控制客户端间分布

# 生成客户端分配比例（Dirichlet分布）
np.random.seed(42)
client_distributions = {}
for cls in class_names:
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    client_distributions[cls] = proportions

# 分配数据到客户端（保存元组 (class_name, index)）
client_data = defaultdict(list)
for cls in class_names:
    indices = balanced_indices[cls]
    proportions = client_distributions[cls]
    client_samples = (proportions * len(indices)).astype(int)
    client_samples[-1] = len(indices) - sum(client_samples[:-1])
    print(client_samples)
    np.random.shuffle(indices)
    ptr = 0
    for client_id in range(num_clients):
        client_data[client_id].extend(indices[ptr:ptr+client_samples[client_id]])
        ptr += client_samples[client_id]

# 验证客户端内部分布（统计class_name）
for client_id in range(num_clients):
    print(f"\nClient {client_id} 的样本分布（总数: {len(client_data[client_id])}）:")
    cls_counts = defaultdict(int)
    for sample in client_data[client_id]:
        cls_counts[sample[0]] += 1  # sample[0] 是 class_name
    for cls in class_names:
        print(f"{cls}: {cls_counts[cls]}", end=" | ")
