import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageStat
import os
import h5py
from PIL import Image
from torchvision import transforms
import joint_transforms

Audio_path = "F:\\wgt\\AVE\\AVE_Dataset\\audio2\\"
Fixat_path = 'F:\\wgt\\AVE\\AVE_Dataset\\Img2\\'
def make_dataset(ori_path):
    path_list = []
    sequ = -1
    ori_name = os.listdir(ori_path)
    for file in range(0, len(ori_name)):
        print(file)
        ficpath = os.path.join(ori_path, ori_name[file])
        ficname = os.listdir(ficpath)
        for fs in range(0, len(ficname)):
            picpath = os.path.join(ficpath, ficname[fs])
            picname = os.listdir(picpath)
            sequ = sequ + 1
            if len(picname)>5:
                for picp in range(0, len(picname)):
                    if picname[picp].endswith('_c.jpg'):
                        ps = os.path.join(picpath, picname[picp])
                        pv = os.path.join(Fixat_path, ori_name[file], ficname[fs], picname[picp][:-6]+'.jpg')
                        pa = os.path.join(Audio_path, ori_name[file], ficname[fs], picname[picp][:-6]+'_asp.h5')
                        path_list.append(pa+'+'+pv+'+'+ps+'+'+str(file)+'+'+ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-6]+'.jpg')
    return path_list

class ImageFolder(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transforms.Compose([
            joint_transforms.RandomCrop(356),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomRotate(10)
        ])
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_transform = transforms.ToTensor()

    def __getitem__(self, index):
        pathimla = self.imgs[index]
        img_la = pathimla.split('+')
        audio_path = img_la[0]
        video_path = img_la[1]
        fixat_path = img_la[2]
        with h5py.File(audio_path, 'r') as hf:
            audio_features = np.float32(hf['dataset'][:])  # 5,128
        Img = Image.open(video_path).resize(
            (400, 400), Image.ANTIALIAS).convert('RGB')
        fixat = Image.open(fixat_path).resize((400, 400)).convert('L')
        ind = int(img_la[3])
        file = img_la[4]
        subfile = img_la[5]
        ssubfile = img_la[6]
        audio_features_batch = audio_features
        
        Img, fixat = self.joint_transform(Img, fixat)
        Img = self.img_transform(Img)
        fixat = self.target_transform(fixat)

        return torch.from_numpy(audio_features_batch).float(), Img, fixat, ind, file, subfile, ssubfile

    def __len__(self):
        return len(self.imgs)
