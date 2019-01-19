import os
import random
import torch
import torch.utils.data as data
import torchvision.transforms.functional as tF
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from PIL import Image
import pandas as pd
import numpy as np



def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


class ImageFolder(data.Dataset):

    def __init__(self, root="../../input",
                 csv_file="train.csv", SEED=1234,
                 transform=None,
                 loader=pil_loader, eval=False, Full=False,
                 TTA=False, n_splits=5,
                 iteration=0):

        # Devide by KFold
        self.data = pd.read_csv(os.path.join(root, csv_file)).loc[:100, :]
        self.data["masks"] = [np.asarray(loader(os.path.join(root, "train/masks",
                                                  self.data.loc[i, "id"]+".png"))) / 255
                              for i in self.data.index]
        self.data["coverage"] = self.data.masks.map(np.sum) / pow(101,2)
        self.data["coverage_class"] = self.data.coverage.map(cov_to_class)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=SEED)

        for i,(dev_index, val_index) in enumerate(skf.split(self.data, self.data["coverage_class"])):
            if i == iteration:
                if not eval:
                    self.data = self.data.loc[dev_index,:]
                else:
                    self.data = self.data.loc[val_index,:]


        self.data.reset_index(inplace=True)
        self.root = root
        self.transform = transform
        self.loader = loader
        self.TTA = TTA
        # del data
        # gc.collect()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.data.loc[index]
        file_name = item["id"] + ".png"
        img = self.loader(os.path.join(self.root, "train/images", file_name))
        mask = self.loader(os.path.join(self.root, "train/masks", file_name))

        SEED = np.random.randint(2147483647)
        if self.TTA:
            img_flip = tF.hflip(img)
            random.seed(SEED)
            mask_flip = tF.hflip(mask)
            img = self.transform(img)
            random.seed(SEED)
            img_flip = self.transform(img_flip)
            random.seed(SEED)
            mask = self.transform(mask)
            random.seed(SEED)
            mask_flip = self.transform(mask_flip)

            return img, img_flip, mask, mask_flip

        else:
            random.seed(SEED)
            img = self.transform(img)
            random.seed(SEED)
            mask = self.transform(mask)
            return img, mask

        # img = new[0]
        # mask = new[1]

    def __len__(self):
        return len(self.data)


class SubmitFolder(data.Dataset):

    def __init__(self, root="../../input",
                 csv_file="sample_submission.csv",
                 transform=None,
                 loader=pil_loader,
                 TTA=False,
                 TTA_2 = False):

        self.data = pd.read_csv(os.path.join(root, csv_file))#.iloc[:100]
        self.data.reset_index(inplace=True)
        self.root = root
        self.transform = transform
        self.loader = loader
        self.TTA = TTA
        self.TTA_2 = TTA_2

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.data.loc[index]
        file_name = item["id"] + ".png"
        # target = item['rle_mask']
        img = self.loader(os.path.join(self.root, "images", file_name))
        # print("img shape : ", img.size)
        # print("mask shape : ", mask.size)
        # new = torch.cat((img, mask))

        if self.TTA:
            img_flip = tF.hflip(img)
            img = self.transform(img)
            img_flip = self.transform(img_flip)
            return img, img_flip, item["id"]

        elif self.TTA_2:
            img_flip_v = tF.vflip(img)
            img_flip_h = tF.hflip(img)
            img = self.transform(img)
            img_flip_v = self.transform(img_flip_v)
            img_flip_h = self.transform(img_flip_h)
            return img, img_flip_h, img_flip_v, item["id"]

        else:
            img = self.transform(img)
            return img, item["id"]


    def __len__(self):
        return len(self.data)



class ImageFolder_supervision(data.Dataset):

    def __init__(self, root="../../input",
                 csv_file="train.csv", SEED=1234,
                 transform=None,
                 loader=pil_loader, eval=False, Full=False,
                 TTA=False, n_splits=5,
                 iteration=0):

        # Devide by KFold
        self.data = pd.read_csv(os.path.join(root, csv_file))#.loc[:100, :]
        self.data["masks"] = [np.asarray(loader(os.path.join(root, "train/masks",
                                                  self.data.loc[i, "id"]+".png"))) / 255
                              for i in self.data.index]
        self.data["coverage"] = self.data.masks.map(np.sum) / pow(101,2)
        self.data["coverage_class"] = self.data.coverage.map(cov_to_class)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=SEED)

        for i,(dev_index, val_index) in enumerate(skf.split(self.data, self.data["coverage_class"])):
            if i == iteration:
                if not eval:
                    self.data = self.data.loc[dev_index,:]
                else:
                    self.data = self.data.loc[val_index,:]


        self.data.reset_index(inplace=True)
        self.root = root
        self.transform = transform
        self.loader = loader
        self.TTA = TTA
        # del data
        # gc.collect()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.data.loc[index]
        file_name = item["id"] + ".png"
        img = self.loader(os.path.join(self.root, "train/images", file_name))
        mask = self.loader(os.path.join(self.root, "train/masks", file_name))
        if np.asarray(mask).sum() == 0:
            truth_image = torch.Tensor([0])
        else:
            truth_image = torch.Tensor([1])

        SEED = np.random.randint(2147483647)
        random.seed(SEED)
        img = self.transform(img)
        random.seed(SEED)
        mask = self.transform(mask)
        return img, mask, truth_image




import torch
import torchvision.transforms as transforms

if __name__=="__main__":
    import numpy as np
    x = torch.randn(2, 1, 101, 101)
    y = torch.randn(2, 1, 101, 101)
    new = torch.stack((x,y), 0)
    print(new.shape)
    print(new[0].shape)
    print(new[1].shape)

    train_loader = torch.utils.data.DataLoader(
        ImageFolder(
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(128, scale=(0.9, 1.0)),  # ratio=(1,1)),
                # transforms.Resize(128),
                transforms.ToTensor()
            ]),
            eval=False, iteration=0, n_splits=5,
        ),
        batch_size=1, shuffle=True,
        num_workers=1, pin_memory=False, drop_last=True)
    print(len(train_loader))
    for i, (img, mask) in enumerate(train_loader):
        if i >10:
            break
        print(img.size)
        print(mask.size)
        print(np.asarray(mask))
        print(np.logical_and(np.asarray(img), np.asarray(mask)))
        print(np.logical_or(np.random.randn(101,101), np.asarray(mask)))
