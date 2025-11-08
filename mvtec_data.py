import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

class MVTECADataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.images, self.labels = self.load_data()

    def load_data(self):
        images = []
        labels = []
        categories = os.listdir(self.root_dir)
        category_to_idx = {category: idx for idx, category in enumerate(categories)}
        for category in categories:
            category_dir = os.path.join(self.root_dir, category, self.subset)
            if not os.path.isdir(category_dir):
                continue
            for root, _, files in os.walk(category_dir):
                for file in files:
                    if file.endswith('.png') or file.endswith('.jpg'):
                        images.append(os.path.join(root, file))
                        labels.append(category_to_idx[category])
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_data_loaders(self, batch_size=32, shuffle=True):
        data_loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        return data_loader

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

if __name__ == "__main__":
    from torchvision import transforms

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Initialize datasets
    train_dataset = MVTECADataset(root_dir='./Folder/Data/Raw/mvtec_data', subset='train', transform=transform)
    test_dataset = MVTECADataset(root_dir='./Folder/Data/Raw/mvtec_data', subset='test', transform=transform)

    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Test dataset length: {len(test_dataset)}')

    # Get data loaders
    train_loader = train_dataset.get_data_loaders(batch_size=32, shuffle=True)
    test_loader = test_dataset.get_data_loaders(batch_size=32, shuffle=True)

    # Print the shape of one batch from train loader
    for images, labels in train_loader:
        print(f'Train batch images shape: {images.shape}')
        print(f'Train batch labels: {labels}')
        break

    # Print the shape of one batch from test loader
    for images, labels in test_loader:
        print(f'Test batch images shape: {images.shape}')
        print(f'Test batch labels: {labels}')
        break