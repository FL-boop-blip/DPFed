import numpy as np
from collections import defaultdict
class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
class_counts = [1441, 545, 796, 5965, 2475, 1615, 2502]
class_indices = {}

for i, name in enumerate(class_names):
    class_indices[name] = list(range(sum(class_counts[:i]), sum(class_counts[:i+1])))
# print(class_indices)
np.random.seed(42)  # For reproducibility
num_clients = 10
alpha = 100000  # Dirichlet parameter
client_distributions = {}
for cls in class_names:
    proportions = np.random.dirichlet(np.repeat(alpha,num_clients))
    print(f"Proportions for {cls}: {proportions}")
    client_distributions[cls] = proportions

client_data = defaultdict(list)
# print(client_data)
for cls, indices in class_indices.items():
    proportions = client_distributions[cls]
    num_samples = len(indices)
    client_samples = (proportions * num_samples).astype(int)
    print(client_samples)
    client_samples[-1] = num_samples - sum(client_samples[:-1])  # Ensure the last client gets the remaining samples
    np.random.shuffle(indices)  # Shuffle indices for randomness
    ptr = 0
    for client_id in range(num_clients):
        num = client_samples[client_id]
        client_data[client_id].extend(indices[ptr:ptr + num])
        ptr += num

for client_id in range(num_clients):
    print(f"\nClient {client_id} 的样本分布:")
    for cls in class_names:
        cls_count = len(set(client_data[client_id]) & set(class_indices[cls]))
        print(f"{cls}: {cls_count}", end=" | ")