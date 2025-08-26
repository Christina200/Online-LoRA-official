import os
import torch
from torchvision.datasets.utils import download_url
import tqdm
from .continual_datasets import *

class CORe50(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'
        self.filename = 'core50_128x128.zip'

        # self.fpath = os.path.join(root, 'core50_128x128')
        # if not os.path.isfile(self.fpath):
        #     if not download:
        #        raise RuntimeError('Dataset not found. You can use download=True to download it')
        #     else:
        #         print('Downloading from '+self.url)
        #         download_url(self.url, root, filename=self.filename)

        # if not os.path.exists(os.path.join(root, 'core50_128x128')):
        #     with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zf:
        #         for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
        #             try:
        #                 zf.extract(member, root)
        #             except zipfile.error as e:
        #                 pass

        self.fpath = '/home/xw6956/Si-Blurry/local_datasets/core50_128x128'

        self.train_session_list = ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11']
        self.test_session_list = ['s3', 's7', 's10']
        self.label = [f'o{i}' for i in range(1, 51)]
        self.targets = []
        
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.split()
        
        if self.train:
            fpath = self.fpath + '/train'
            self.dataset = [datasets.ImageFolder(f'{fpath}/{s}', transform=transform) for s in self.train_session_list]
            for d in self.dataset:
                self.targets.extend(d.targets)
        else:
            fpath = self.fpath + '/test'
            self.dataset = datasets.ImageFolder(fpath, transform=transform)
            self.targets = self.dataset.targets

        self.classes = [str(i) for i in range(50)]
        


    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

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