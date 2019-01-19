import os
import pandas as pd
from PIL import Image
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ImageFolder_cla(data.Dataset):

    def __init__(self, root, csv_file, transform=None, target_transform=None,
                 loader=pil_loader, iteration=0, mode="train", n_splits=5, SEED=519):
        self.data = pd.read_csv(csv_file)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=SEED)
        for i, (dev_index, val_index) in enumerate(skf.split(self.data, self.data["class"])):
            if i == iteration:
                if mode == "train":
                    self.data = self.data.loc[dev_index, :]
                elif mode == "val":
                    self.data = self.data.loc[val_index, :]
                else:
                    raise ValueError("Wrong mode")
        self.data.reset_index(inplace=True)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.data.loc[index]
        folder = item['folder_name']
        file_name = item["file_name"]
        target = item['category_id']

        img = self.loader(os.path.join(self.root, "images", folder, file_name))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class SubmitFolder(data.Dataset):

    def __init__(self, root="PATH",
                 csv_file="test_submission.csv",
                 transform=None,
                 loader=pil_loader,
                 ):

        self.data = pd.read_csv(os.path.join(root, csv_file))#.loc[:512, :]
        self.root = root
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.data.loc[index]
        file_name = item["image"]
        directory = item["directory"]
        img = self.loader(os.path.join(self.root, directory, file_name))

        if self.transform is not None:
            img = self.transform(img)

        return img, file_name

    def __len__(self):
        return len(self.data)