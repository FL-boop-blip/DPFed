import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from torchvision.transforms import transforms
import os
from PIL import Image, ImageOps
import random

class RAFDB(Dataset):
    def __init__(self, root_dir, sub_set='train', transform=None):
        self.root_dir = root_dir
        self.subset = sub_set
        self.transform = transform
        self.img_info = []  # [(path,label), ...,]
        self.label_array = None
        self._emotion_to_categories = {
            'Surprise': 0,
            'Fear': 1,
            'Disgust': 2,
            'Happiness': 3,
            'Sadness': 4,
            'Anger': 5,
            'Neutral': 6
        }
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir: {} is a empty dir! Please checkout your path to images!".format(self.root_dir))
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    path_img = os.path.join(root, file)
                    sub_dir = os.path.basename(root)
                    # print(sub_dir)
                    label_int = self._emotion_to_categories[sub_dir]
                    # print(label_int,sub_dir)
                    self.img_info.append((path_img, label_int))
    


    # def show_category(self):
    #     print(self.img_info[:50])
    #     print(self.categories)




if __name__ == '__main__':
    root_dir_train = r"../Data/Raw/RAF/train"
    root_dir_test = r"../Data/Raw/RAF/valid"
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((64, 64)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    train_set = RAFDB(root_dir_train, transform=transform)
    test_set = RAFDB(root_dir_test, transform=transform)
    print(len(train_set))
    print(len(test_set))

    # print(train_set.show_category())

    train_loader = DataLoader(dataset=train_set,batch_size=50,shuffle=True)
    for i, (inputs, targets) in enumerate(train_loader):
        print(f'inputs shape is {inputs.shape}, targets shape is {targets.shape}')
