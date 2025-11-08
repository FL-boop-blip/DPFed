import os
import numpy as np
import io
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from Folder.data_process import Fer2013, Fer2013_plus, RAFDB, CELEA
from torch.utils.data import ConcatDataset

data_path = "Folder/"
class DatasetObject:
    def __init__(self, dataset, n_clnt, seed, rule, unbalanced_sgm=0, rule_arg='', data_path=''):
        self.dataset = dataset
        self.n_clnt = n_clnt
        self.rule = rule
        self.rule_arg = rule_arg
        self.seed = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        # self.name = "{:s}_{:s}_{:s}_{:.0f}%-{:d}".format(dataset, rule, str(rule_arg), args.active_ratio*args.total_client, args.total_client)
        self.name = "%s_%d_%d_%s_%s" % (self.dataset, self.n_clnt, self.seed, self.rule, rule_arg_str)
        self.name += '_%f' % unbalanced_sgm if unbalanced_sgm != 0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.set_data()

    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%sData/%s' % (self.data_path, self.name)):
            # Get Raw data
            if self.dataset == 'mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trnset = torchvision.datasets.MNIST(root='%sData/Raw' % self.data_path,
                                                    train=True, download=True, transform=transform)
                tstset = torchvision.datasets.MNIST(root='%sData/Raw' % self.data_path,
                                                    train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;

            if self.dataset == 'SVHN':
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
                    transforms.ToTensor()])

                trnset = torchvision.datasets.SVHN(root='%sData/Raw' % self.data_path,
                                                   split="train", download=True, transform=transform)
                tstset = torchvision.datasets.SVHN(root='%sData/Raw' % self.data_path,
                                                   split="test", download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=75257, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=26032, shuffle=False, num_workers=0)
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 10;

            if self.dataset == 'fer2013':
                root_dir_train = "./Folder/Data/Raw/fer2013/train"
                root_dir_test = "./Folder/Data/Raw/fer2013/test"
                root_dir_val = "./Folder/Data/Raw/fer2013/val"
                transform_trn = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                transform_tst = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                trnset = Fer2013(root_dir_train, transform=transform_trn)
                tstset = Fer2013(root_dir_test, transform=transform_tst)
                valset = Fer2013(root_dir_val, transform=transform_tst)
                combined_set = ConcatDataset([tstset, valset])

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=len(trnset), shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(combined_set, batch_size=len(combined_set), shuffle=False,
                                                       num_workers=0)
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 7;

            if self.dataset == 'fer2013_plus':
                root_dir_train = "./Folder/Data/Raw/fer2013_plus/train"
                root_dir_test = "./Folder/Data/Raw/fer2013_plus/test"
                root_dir_val = "./Folder/Data/Raw/fer2013_plus/val"
                transform_trn = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

                transform_tst = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                trnset = Fer2013_plus(root_dir_train, transform=transform_trn)
                tstset = Fer2013_plus(root_dir_test, transform=transform_tst)
                valset = Fer2013_plus(root_dir_val, transform=transform_tst)
                combined_set = ConcatDataset([tstset, valset])

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=len(trnset), shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(combined_set, batch_size=len(combined_set), shuffle=False,
                                                       num_workers=0)
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 8;

            if self.dataset == 'rafdb':
                root_dir_train = "./Folder/Data/Raw/RAF/train"
                root_dir_test = "./Folder/Data/Raw/RAF/valid"
                transform_trn = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                transform_tst = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                trnset = RAFDB(root_dir_train, transform=transform_trn)
                tstset = RAFDB(root_dir_test, transform=transform_tst)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=len(trnset), shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=len(tstset), shuffle=False, num_workers=0)
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 7;

            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                                     std=[0.247, 0.243, 0.262])])

                trnset = torchvision.datasets.CIFAR10(root='%sData/Raw' % self.data_path,
                                                      train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='%sData/Raw' % self.data_path,
                                                      train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 10;

            if self.dataset == 'CIFAR100':
                print(self.dataset)
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trnset = torchvision.datasets.CIFAR100(root='%sData/Raw' % self.data_path,
                                                       train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR100(root='%sData/Raw' % self.data_path,
                                                       train=False, download=True, transform=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 100;

            if self.dataset == 'celea':
                print(self.dataset)
                root_dir_train = "./Folder/Data/Raw/celeba/train"
                root_dir_test = "./Folder/Data/Raw/celeba/test"
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((64, 64)),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trnset = CELEA(root_dir=root_dir_train, transform=transform)
                tstset = CELEA(root_dir=root_dir_test, transform=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=40000, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=8000, shuffle=False, num_workers=0)
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 4;

            if self.dataset == 'tinyimagenet':
                print(self.dataset)
                transform = transforms.Compose([  # transforms.Resize(224),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], #pre-train
                    #                      std=[0.229, 0.224, 0.225])])
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])])
                # trainset = torchvision.datasets.ImageFolder(root='%sData/Raw' %self.data_path,
                #                                       train=True , download=True, transform=transform)
                # testset = torchvision.datasets.ImageFolder(root='%sData/Raw' %self.data_path,
                #                                       train=False, download=True, transform=transform)
                # root_dir = self.data_path
                root_dir = "./Data/Raw/tiny-imagenet-200/"
                trn_img_list, trn_lbl_list, tst_img_list, tst_lbl_list = [], [], [], []
                trn_file = os.path.join(root_dir, 'train_list.txt')
                tst_file = os.path.join(root_dir, 'val_list.txt')
                with open(trn_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        trn_img_list.append(img)
                        trn_lbl_list.append(int(lbl))
                with open(tst_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        tst_img_list.append(img)
                        tst_lbl_list.append(int(lbl))
                trnset = DatasetFromDir(img_root=root_dir, img_list=trn_img_list, label_list=trn_lbl_list,
                                        transformer=transform)
                tstset = DatasetFromDir(img_root=root_dir, img_list=tst_img_list, label_list=tst_lbl_list,
                                        transformer=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=len(trnset), shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=len(tstset), shuffle=False, num_workers=0)
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 200;

            if self.dataset != 'emnist':
                trn_itr = trn_load.__iter__();
                tst_itr = tst_load.__iter__()
                # labels are of shape (n_data,)
                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()

                trn_x = trn_x.numpy();
                trn_y = trn_y.numpy().reshape(-1, 1)
                tst_x = tst_x.numpy();
                tst_y = tst_y.numpy().reshape(-1, 1)
                # 这里的trn_x是一个batch的大小，trn_y是一个batch的标签，而batch是整个数据集的大小

            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "Data/Raw/matlab/emnist-letters.mat")
                # load training dataset
                x_trn = emnist["dataset"][0][0][0][0][0][0]
                x_trn = x_trn.astype(np.float32)

                # load training labels
                y_trn = emnist["dataset"][0][0][0][0][0][1] - 1  # make first class 0

                # take first 10 classes of letters
                trn_idx = np.where(y_trn < 10)[0]

                y_trn = y_trn[trn_idx]
                x_trn = x_trn[trn_idx]

                mean_x = np.mean(x_trn)
                std_x = np.std(x_trn)

                # load test dataset
                x_tst = emnist["dataset"][0][0][1][0][0][0]
                x_tst = x_tst.astype(np.float32)

                # load test labels
                y_tst = emnist["dataset"][0][0][1][0][0][1] - 1  # make first class 0

                tst_idx = np.where(y_tst < 10)[0]

                y_tst = y_tst[tst_idx]
                x_tst = x_tst[tst_idx]

                x_trn = x_trn.reshape((-1, 1, 28, 28))
                x_tst = x_tst.reshape((-1, 1, 28, 28))

                # normalise train and test features

                trn_x = (x_trn - mean_x) / std_x
                trn_y = y_trn

                tst_x = (x_tst - mean_x) / std_x
                tst_y = y_tst

                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;

            # Shuffle Data，将训练集的数据打乱，以CIFAR10为例，生成0-50000的随机索引
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]

            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y

            ###CIAFR10,每个客户端分配的数据量是一样的，100个客户端的话，每个客户端分配500个数据
            n_data_per_clnt = int((len(trn_y)) / self.n_clnt)
            # Draw from lognormal distribution
            #  初始化每个客户端的数据量
            clnt_data_list = np.ones(self.n_clnt, dtype=int) * n_data_per_clnt
            diff = np.sum(clnt_data_list) - len(trn_y)

            # Add/Subtract the excess number starting from first client
            if diff != 0:
                for clnt_i in range(self.n_clnt):
                    if clnt_data_list[clnt_i] > diff:
                        clnt_data_list[clnt_i] -= diff
                        break
            ###

            if self.rule == 'Dirichlet' or self.rule == 'Pathological':
                if self.rule == 'Dirichlet':
                    cls_priors = np.random.dirichlet(alpha=[self.rule_arg] * self.n_cls, size=self.n_clnt)
                    # np.save("results/heterogeneity_distribution_{:s}.npy".format(self.dataset), cls_priors)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                elif self.rule == 'Pathological':
                    c = int(self.rule_arg)
                    a = np.ones([self.n_clnt, self.n_cls])
                    a[:, c::] = 0
                    [np.random.shuffle(i) for i in a]
                    # np.save("results/heterogeneity_distribution_{:s}_{:s}.npy".format(self.dataset, self.rule), a/c)
                    prior_cumsum = a.copy()
                    for i in range(prior_cumsum.shape[0]):
                        for j in range(prior_cumsum.shape[1]):
                            if prior_cumsum[i, j] != 0:
                                prior_cumsum[i, j] = a[i, 0:j + 1].sum() / c * 1.0

                idx_list = [np.where(trn_y == i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                true_sample = [0 for i in range(self.n_cls)]
                # print(cls_amount)
                clnt_x = [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                          for clnt__ in range(self.n_clnt)]
                clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_clnt)]

                while (np.sum(clnt_data_list) != 0):
                    # 随机选取一个客户端
                    curr_clnt = np.random.randint(self.n_clnt)
                    # If current node is full resample a client
                    # print('Remaining Data: %d' %np.sum(client_data_list))
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if train_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            cls_amount[cls_label] = len(idx_list[cls_label])
                            continue
                        cls_amount[cls_label] -= 1
                        true_sample[cls_label] += 1

                        clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break
                print(true_sample)
                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

                # clnt_x = np.asarray(clnt_x,dtype=object)
                # clnt_y = np.asarray(clnt_y,dtype=object)


            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm == 0:
                assert len(trn_y) // 100 % self.n_clnt == 0

                # create perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(trn_y[:, 0])
                n_data_per_clnt = len(trn_y) // self.n_clnt
                clnt_x = np.zeros((self.n_clnt, n_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y = np.zeros((self.n_clnt, n_data_per_clnt, 1), dtype=np.float32)
                trn_x = trn_x[idx]  # 50000*3*32*32
                trn_y = trn_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 100
                for i in range(self.n_clnt):  # devices
                    for j in range(100):  # class
                        clnt_x[i, n_cls_sample_per_device * j:n_cls_sample_per_device * (j + 1), :, :, :] = trn_x[
                                                                                                            500 * j + n_cls_sample_per_device * i:500 * j + n_cls_sample_per_device * (
                                                                                                                        i + 1),
                                                                                                            :, :, :]
                        clnt_y[i, n_cls_sample_per_device * j:n_cls_sample_per_device * (j + 1), :] = trn_y[
                                                                                                      500 * j + n_cls_sample_per_device * i:500 * j + n_cls_sample_per_device * (
                                                                                                                  i + 1),
                                                                                                      :]


            elif self.rule == 'iid':

                clnt_x = [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                          for clnt__ in range(self.n_clnt)]
                clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_clnt)]

                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_clnt):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_ + 1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_ + 1]]

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

                # clnt_x = np.asarray(clnt_x,dtype=object)
                # clnt_y = np.asarray(clnt_y,dtype=object)

            self.clnt_x = clnt_x;
            self.clnt_y = clnt_y

            self.tst_x = tst_x;
            self.tst_y = tst_y

            # Save data
            print('begin to save data...')
            os.mkdir('%sData/%s' % (self.data_path, self.name))

            np.save('%sData/%s/clnt_x.npy' % (self.data_path, self.name), clnt_x)
            np.save('%sData/%s/clnt_y.npy' % (self.data_path, self.name), clnt_y)

            np.save('%sData/%s/tst_x.npy' % (self.data_path, self.name), tst_x)
            np.save('%sData/%s/tst_y.npy' % (self.data_path, self.name), tst_y)

            print('data loading finished.')

        else:
            print("Data is already downloaded")
            # self.clnt_x = np.load('%sData/%s/clnt_x.npy' %(self.data_path, self.name), mmap_mode = 'r')
            self.clnt_x = np.load('%sData/%s/clnt_x.npy' % (self.data_path, self.name), allow_pickle=True)
            self.clnt_y = np.load('%sData/%s/clnt_y.npy' % (self.data_path, self.name), allow_pickle=True)
            self.n_clnt = len(self.clnt_x)

            self.tst_x = np.load('%sData/%s/tst_x.npy' % (self.data_path, self.name), allow_pickle=True)
            self.tst_y = np.load('%sData/%s/tst_y.npy' % (self.data_path, self.name), allow_pickle=True)

            if self.dataset == 'mnist':
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;
            if self.dataset == 'SVHN':
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 10;
            if self.dataset == "fer2013":
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 7;
            if self.dataset == "fer2013_plus":
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 8;
            if self.dataset == 'rafdb':
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 7;
            if self.dataset == 'CIFAR10':
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 100;
            if self.dataset == 'emnist':
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;
            if self.dataset == 'celea':
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 4;
            if self.dataset == 'tinyimagenet':
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 200;

            print('data loading finished.')
        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_clnt):
            print("Client %3d: " % clnt +
                  ', '.join(["%.3f" % np.mean(self.clnt_y[clnt] == cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' % self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]

        print('Total Amount:%d' % count)
        print('--------')

        print("      Test: " +
              ', '.join(["%.3f" % np.mean(self.tst_y == cls) for cls in range(self.n_cls)]) +
              ', Amount:%d' % self.tst_y.shape[0])


def generate_syn_logistic(dimension, n_clnt, n_cls, avg_data=4, alpha=1.0, beta=0.0, theta=0.0, iid_sol=False,
                          iid_dat=False):
    # alpha is for minimizer of each client
    # beta  is for distirbution of points
    # theta is for number of data points

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    samples_per_user = (np.random.lognormal(mean=np.log(avg_data + 1e-3), sigma=theta, size=n_clnt)).astype(int)
    print('samples per user')
    print(samples_per_user)
    print('sum %d' % np.sum(samples_per_user))

    num_samples = np.sum(samples_per_user)

    data_x = list(range(n_clnt))
    data_y = list(range(n_clnt))

    mean_W = np.random.normal(0, alpha, n_clnt)
    B = np.random.normal(0, beta, n_clnt)

    mean_x = np.zeros((n_clnt, dimension))

    if not iid_dat:  # If IID then make all 0s.
        for i in range(n_clnt):
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    sol_W = np.random.normal(mean_W[0], 1, (dimension, n_cls))
    sol_B = np.random.normal(mean_W[0], 1, (1, n_cls))

    if iid_sol:  # Then make vectors come from 0 mean distribution
        sol_W = np.random.normal(0, 1, (dimension, n_cls))
        sol_B = np.random.normal(0, 1, (1, n_cls))

    for i in range(n_clnt):
        if not iid_sol:
            sol_W = np.random.normal(mean_W[i], 1, (dimension, n_cls))
            sol_B = np.random.normal(mean_W[i], 1, (1, n_cls))

        data_x[i] = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        data_y[i] = np.argmax((np.matmul(data_x[i], sol_W) + sol_B), axis=1).reshape(-1, 1)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    return data_x, data_y


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'mnist' or self.name == 'emnist' or self.name == 'SVHN':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()

        elif self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name == "tinyimagenet" or self.name == 'fer2013' or self.name == 'fer2013_plus' or self.name == 'rafdb' or self.name == 'celea':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])

            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')

        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'mnist' or self.name == 'emnist' or self.name == 'SVHN':
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y

        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
                if (np.random.rand() > .5):
                    # Random cropping
                    pad = 4
                    extended_img = np.zeros((3, 32 + pad * 2, 32 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 32, dim_2:dim_2 + 32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

        elif self.name == 'tinyimagenet' or self.name == 'fer2013' or self.name == 'fer2013_plus' or self.name == 'rafdb' or self.name == 'celea':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
                if np.random.rand() > .5:
                    # Random cropping
                    pad = 8
                    extended_img = np.zeros((3, 64 + pad * 2, 64 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 64, dim_2:dim_2 + 64]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

        else:
            raise NotImplementedError


class DatasetFromDir(data.Dataset):

    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        # ********************
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]

        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    dataset = DatasetObject(dataset='fer2013_plus', n_clnt=100, seed=0, rule='iid',data_path=data_path)
    print(dataset.clnt_x.shape)
    print(dataset.clnt_y.shape)
    print(dataset.tst_x.shape)
    print(dataset.tst_y.shape)

    # data_x, data_y = generate_syn_logistic(100, 10, 2, iid_sol=True, iid_dat=False)
    # dataset = Dataset(data_x, data_y, dataset_name='logistic')
