# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import os
import os.path
import warnings
import pathlib
from pathlib import Path

from typing import Any, Tuple

import glob
from shutil import move, rmtree

import numpy as np

import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive

import PIL
from PIL import Image

import tqdm
import zipfile
import tarfile

from .dataset_utils import read_image_file, read_label_file

class MNIST_RGB(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST_RGB, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FashionMNIST(MNIST_RGB):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class NotMNIST(MNIST_RGB):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip'
        self.filename = 'notMNIST.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'notMNIST')):
            with zipfile.ZipFile(fpath, 'r') as zf:
                for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
                    try:
                        zf.extract(member, root)
                    except zipfile.error as e:
                        pass
        if self.train:
            fpath = os.path.join(root, 'notMNIST', 'Train')

        else:
            fpath = os.path.join(root, 'notMNIST', 'Test')


        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img)
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class PermutedMNIST(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, random_seed=None):
        super(PermutedMNIST, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.train = train  # training set or test set 
        self.random_seed = random_seed

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # Permutation
        size = img.size(0)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        perm = np.random.permutation(size*size)
        img = img.flatten()[perm].reshape(size, size)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTM(torch.utils.data.Dataset):
    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = os.path.join(root, 'MNIST-M')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.classes = [i for i in range(10)]

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

class SynDigit(torch.utils.data.Dataset):
    resources = [
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_train.pt.gz',
         'd0e99daf379597e57448a89fc37ae5cf'),
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_test.pt.gz',
         '669d94c04d1c91552103e9aded0ee625')
    ]

    training_file = "synth_train.pt"
    test_file = "synth_test.pt"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = os.path.join(root, 'SynDigit')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.classes = [i for i in range(10)]

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.classes = np.unique(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

class Flowers102(datasets.Flowers102):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(Flowers102, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))

        self.targets = []
        self._image_files = []
        for image_id in image_ids:
            self.targets.append(image_id_to_label[image_id] - 1) # -1 for 0-based indexing
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")
        self.classes = list(set(self.targets))
    
    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self.targets[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)

class StanfordCars(datasets.StanfordCars):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super(StanfordCars, self).__init__(root, transform=transform, target_transform=target_transform, download=download)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()

class CUB200(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://data.deepai.org/CUB200(2011).zip'
        self.filename = 'CUB200(2011).zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'CUB_200_2011')):
            with zipfile.ZipFile(fpath, 'r') as zf:
                for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
                    try:
                        zf.extract(member, root)
                    except zipfile.error as e:
                        pass

            with tarfile.open(os.path.join(root, 'CUB_200_2011.tgz'), 'r') as tf:
                for member in tqdm.tqdm(iterable=tf.getmembers(), total=len(tf.getmembers()), desc='Extracting CUB_200_2011.tgz'):
                    try:
                        tf.extract(member, root)
                    except tarfile.TarError as e:
                        pass
            self.split()
        
        if self.train:
            fpath = os.path.join(root, 'CUB_200_2011', 'train')

        else:
            fpath = os.path.join(root, 'CUB_200_2011', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.root + 'CUB_200_2011/train'
        test_folder = self.root + 'CUB_200_2011/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        images = self.root + 'CUB_200_2011/images.txt'
        train_test_split = self.root + 'CUB_200_2011/train_test_split.txt'

        with open(images, 'r') as image:
            with open(train_test_split, 'r') as f:
                for line in tqdm.tqdm(f, desc='Preprocessing'):
                    image_path = image.readline().split(' ')[-1]
                    image_path = image_path.replace('\n', '')
                    class_name = image_path.split('/')[0].split(' ')[-1]
                    src = self.root + 'CUB_200_2011/images/' + image_path

                    if line.split(' ')[-1].replace('\n', '') == '1':
                        if not os.path.exists(train_folder + '/' + class_name):
                            os.mkdir(train_folder + '/' + class_name)
                        dst = train_folder + '/' + image_path
                    else:
                        if not os.path.exists(test_folder + '/' + class_name):
                            os.mkdir(test_folder + '/' + class_name)
                        dst = test_folder + '/' + image_path
                    
                    move(src, dst)

class TinyImagenet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        self.filename = 'tiny-imagenet-200.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)
        
        if not os.path.exists(os.path.join(root, 'tiny-imagenet-200')):
            with zipfile.ZipFile(fpath, 'r') as zf:
                for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
                    try:
                        zf.extract(member, root)
                    except zipfile.error as e:
                        pass
            self.split()

        if self.train:
            fpath = root + 'tiny-imagenet-200/train'

        else:
            fpath = root + 'tiny-imagenet-200/test'
        
        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        test_folder = self.root + 'tiny-imagenet-200/test'

        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(test_folder)

        val_dict = {}
        with open(self.root + 'tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]
                
        paths = glob.glob(self.root + 'tiny-imagenet-200/val/images/*')
        for path in paths:
            if '\\' in path:
                path = path.replace('\\', '/')
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(test_folder + '/' + folder):
                os.mkdir(test_folder + '/' + folder)
                os.mkdir(test_folder + '/' + folder + '/images')
            
            
        for path in tqdm.tqdm(paths, desc='Preprocessing'):
            if '\\' in path:
                path = path.replace('\\', '/')
            file = path.split('/')[-1]
            folder = val_dict[file]
            src = path
            dst = test_folder + '/' + folder + '/images/' + file
            move(src, dst)
        
        rmtree(self.root + 'tiny-imagenet-200/val')

class Scene67(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        image_url = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
        train_annos_url = 'http://web.mit.edu/torralba/www/TrainImages.txt'
        test_annos_url = 'http://web.mit.edu/torralba/www/TestImages.txt'
        urls = [image_url, train_annos_url, test_annos_url]
        image_fname = 'indoorCVPR_09.tar'
        self.train_annos_fname = 'TrainImage.txt'
        self.test_annos_fname = 'TestImage.txt'
        fnames = [image_fname, self.train_annos_fname, self.test_annos_fname]

        for url, fname in zip(urls, fnames):
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from ' + url)
                    download_url(url, root, filename=fname)
        if not os.path.exists(os.path.join(root, 'Scene67')):
            with tarfile.open(os.path.join(root, image_fname), 'r') as tf:
                for member in tqdm.tqdm(iterable=tf.getmembers(), total=len(tf.getmembers()), desc=f'Extracting {image_fname}'):
                    try:
                        tf.extract(member, os.path.join(root, 'Scene67'))
                    except tarfile.TarError as e:
                        pass
            self.split()

        if self.train:
            fpath = os.path.join(root, 'Scene67', 'train')

        else:
            fpath = os.path.join(root, 'Scene67', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        if not os.path.exists(os.path.join(self.root, 'Scene67', 'train')):
            os.mkdir(os.path.join(self.root, 'Scene67', 'train'))
        if not os.path.exists(os.path.join(self.root, 'Scene67', 'test')):
            os.mkdir(os.path.join(self.root, 'Scene67', 'test'))
        
        train_annos_file = os.path.join(self.root, self.train_annos_fname)
        test_annos_file = os.path.join(self.root, self.test_annos_fname)

        with open(train_annos_file, 'r') as f:
            for line in tqdm.tqdm(f.readlines(), desc='Preprocessing'):
                line = line.replace('\n', '')
                src = self.root + 'Scene67/' + 'Images/' + line
                dst = self.root + 'Scene67/' + 'train/' + line
                if not os.path.exists(os.path.join(self.root, 'Scene67', 'train', line.split('/')[0])):
                   os.mkdir(os.path.join(self.root, 'Scene67', 'train', line.split('/')[0]))
                move(src, dst)
        
        with open(test_annos_file, 'r') as f:
            for line in tqdm.tqdm(f.readlines(), desc='Preprocessing'):
                line = line.replace('\n', '')
                src = self.root + 'Scene67/' + 'Images/' + line
                dst = self.root + 'Scene67/' + 'test/' + line
                if not os.path.exists(os.path.join(self.root, 'Scene67', 'test', line.split('/')[0])):
                   os.mkdir(os.path.join(self.root, 'Scene67', 'test', line.split('/')[0]))
                move(src, dst)

class Imagenet_R(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
        self.filename = 'imagenet-r.tar'

        self.fpath = os.path.join(root, 'imagenet-r')
        if not os.path.isfile(self.fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'imagenet-r')):
            with tarfile.open(os.path.join(self.root, self.filename), 'r') as tf:
                for member in tqdm.tqdm(iterable=tf.getmembers(), total=len(tf.getmembers()), desc=f'Extracting {self.filename}'):
                    try:
                        tf.extract(member, root)
                    except tarfile.TarError as e:
                        pass
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.dataset = datasets.ImageFolder(self.fpath, transform=transform)
            
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            
            train, val = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            train_idx, val_idx = train.indices, val.indices
    
            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]

            self.split()
        
        if self.train:
            fpath = self.fpath + '/train'

        else:
            fpath = self.fpath + '/test'

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))
        
        for path in tqdm.tqdm(self.train_file_list, desc='Preprocessing'):
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in tqdm.tqdm(self.test_file_list, desc='Preprocessing'):
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)
        
        for c in self.dataset.classes:
            path = os.path.join(self.fpath, c)
            rmtree(path)

class CORe50(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'
        self.filename = 'core50_128x128.zip'

        self.fpath = os.path.join(root, 'core50_128x128')
        if not os.path.isfile(self.fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'core50_128x128')):
            with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zf:
                for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
                    try:
                        zf.extract(member, root)
                    except zipfile.error as e:
                        pass

        self.train_session_list = ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11']
        self.test_session_list = ['s3', 's7', 's10']
        self.label = [f'o{i}' for i in range(1, 51)]
        
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.split()
        
        if self.train:
            fpath = self.fpath + '/train'
            self.data = [datasets.ImageFolder(f'{fpath}/{s}', transform=transform) for s in self.train_session_list]
        else:
            fpath = self.fpath + '/test'
            self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for s in tqdm.tqdm(self.train_session_list, desc='Preprocessing'):
            src = os.path.join(self.fpath, s)
            if os.path.exists(os.path.join(train_folder, s)):
                continue
            move(src, train_folder)
        
        for s in tqdm.tqdm(self.test_session_list, desc='Preprocessing'):
            for l in self.label:
                dst = os.path.join(test_folder, l)
                if not os.path.exists(dst):
                    os.mkdir(os.path.join(test_folder, l))
                
                f = glob.glob(os.path.join(self.fpath, s, l, '*.png'))

                for src in f:
                    move(src, dst)
            rmtree(os.path.join(self.fpath, s))

class DomainNet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = os.path.join(root, 'DomainNet')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip'
        ]

        self.filename = [
            'clipart.zip',
            'infograph.zip',
            'painting.zip',
            'quickdraw.zip',
            'real.zip',
            'sketch.zip'
        ]

        self.train_url_list = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_train.txt'
        ]

        for u in self.train_url_list:
            filename = u.split('/')[-1]
            if not os.path.isfile(os.path.join(self.root, filename)):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+filename)
                    download_url(u, root, filename=filename)
        
        self.test_url_list = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_test.txt'
        ]

        for u in self.test_url_list:
            filename = u.split('/')[-1]
            if not os.path.isfile(os.path.join(self.root, filename)):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+filename)
                    download_url(u, root, filename=filename)

        self.fpath = [os.path.join(self.root, f) for f in self.filename]

        for i in range(len(self.fpath)):
            if not os.path.isfile(self.fpath[i]):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+self.url[i])
                    download_url(self.url[i], root, filename=self.filename[i])

        if not os.path.exists(self.root + '/train') and not os.path.exists(self.root + '/test'):
            for i in range(len(self.fpath)):
                if not os.path.exists(os.path.join(self.root, self.filename[i][:-4])):
                    with zipfile.ZipFile(os.path.join(self.root, self.filename[i]), 'r') as zf:
                        for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename[i]}'):
                            try:
                                zf.extract(member, root)
                            except zipfile.error as e:
                                pass
            
            self.split()

        if self.train:
            fpath = self.root + '/train'
            self.data = [datasets.ImageFolder(f'{fpath}/{d}', transform=transform) for d in os.listdir(fpath)]
        else:
            fpath = self.root + '/test'
            self.data = [datasets.ImageFolder(f'{fpath}/{d}', transform=transform) for d in os.listdir(fpath)]

    def split(self):
        train_folder = self.root + '/train'
        test_folder = self.root + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for i in tqdm.tqdm(range(len(self.train_url_list)), desc='Preprocessing'):
            train_list = self.train_url_list[i].split('/')[-1]
            
            with open(os.path.join(self.root, train_list), 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '')
                    path, _ = line.split(' ')
                    dst = '/'.join(path.split('/')[:2])
                    
                    if not os.path.exists(os.path.join(train_folder, dst)):
                        os.makedirs(os.path.join(train_folder, dst))

                    src = os.path.join(self.root, path)
                    dst = os.path.join(train_folder, path)

                    move(src, dst)
        
        for i in tqdm.tqdm(range(len(self.test_url_list)), desc='Preprocessing'):
            test_list = self.test_url_list[i].split('/')[-1]

            with open(os.path.join(self.root, test_list), 'r') as f:
                for line in f.readlines():
                    line = line.replace('\n', '')
                    path, _ = line.split(' ')
                    dst = '/'.join(path.split('/')[:2])

                    if not os.path.exists(os.path.join(test_folder, dst)):
                        os.makedirs(os.path.join(test_folder, dst))

                    src = os.path.join(self.root, path)
                    dst = os.path.join(test_folder, path)

                    move(src, dst)
            rmtree(os.path.join(self.root, test_list.split('_')[0]))