import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageStat
import os
import h5py
from PIL import Image
from torchvision import transforms

Audio_path = "F:\\wgt\\AVE\\AVE_Dataset\\audio2\\"
Crop_path = 'E:\\crop\\crop\\'
Forg_path = 'F:\\wgt\\AVE\\AVE_Dataset\\forgori\\'
def make_dataset(ori_path):
    path_list = []
    sequ = -1
    count = 0
    ori_name = os.listdir(ori_path)
    for file in range(0, len(ori_name)):
        print(file)
        ficpath = os.path.join(ori_path, ori_name[file])
        ficname = os.listdir(ficpath)
        for fs in range(0, len(ficname)):
            picpath = os.path.join(ficpath, ficname[fs])
            picname = os.listdir(picpath)
            sequ = sequ + 1
            for picp in range(0, len(picname)):
                onoroff = '0'
                if os.path.exists(os.path.join(Forg_path, ori_name[file], ficname[fs], picname[picp][:-4]+'.jpg')):
                    onoroff = '1'
                    pa = os.path.join(Audio_path, ori_name[file], ficname[fs], picname[picp][:-4]+'_asp.h5')
                    path_list.append(onoroff+'+'+pa+'+'+str(file)+'+'+
                                    ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-4]+'.jpg')
    return path_list


class ImageFolder(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = make_dataset(root)

    def __getitem__(self, index):
        pathimla = self.imgs[index]
        img_la = pathimla.split('+')
        onoroff = int(img_la[0])
        audio_path = img_la[1]
        with h5py.File(audio_path, 'r') as hf:
            audio_features = np.float32(hf['dataset'][:])  # 5,128
        audio_features_batch = torch.from_numpy(audio_features).float()
        inda = int(img_la[-4])
        indv = int(img_la[-5])
        file = img_la[-3]
        subfile = img_la[-2]
        ssubfile = img_la[-1]

        return audio_features_batch, onoroff, indv, inda, file, subfile, ssubfile

    def __len__(self):
        return len(self.imgs)
