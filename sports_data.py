import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from torchvision.transforms import transforms

class SportsDataset(Dataset):
    def __init__(self, root_dir, sub_set='train',transform = None):
        self.root_dir = root_dir
        self.subset =sub_set
        self.transform = transform
        self.img_info = [] # [(path,label), ...,]
        self.label_array = None
        self.categories = os.listdir(self.root_dir)
        self.category_to_idx = {category: idx for idx, category in enumerate(self.categories)}
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir: {} is a empty dir! Please checkout your path to images!".format(self.root_dir))
        return len(self.img_info)

    def _get_img_info(self):
        for root,dirs,files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    path_img = os.path.join(root,file)
                    sub_dir = os.path.basename(root)
                    label_int = self.category_to_idx[sub_dir]
                    self.img_info.append((path_img,label_int))

if __name__ == '__main__':
    root_dir_train =r"Folder/Data/Raw/sports_classification/train"
    root_dir_test =r"Folder/Data/Raw/sports_classification/test"
    root_dir_val =r"Folder/Data/Raw/sports_classification/valid"
    normalize = transforms.Normalize([0.5], [0.5])
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_set = SportsDataset(root_dir_train,transform=transforms_train)
    test_set = SportsDataset(root_dir_test,transform=transforms_train)
    valid_set = SportsDataset(root_dir_val,transform=transforms_train)
    combined_set = ConcatDataset([test_set,valid_set])
    print(len(combined_set))

    # train_loader = DataLoader(dataset=train_set,batch_size=50,shuffle=True)
    # for i, (inputs, targets) in enumerate(train_loader):
    #     print(f'inputs shape is {inputs.shape}, targets shape is {targets.shape}')
