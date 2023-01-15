import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch 


class DriveDataset(Dataset):
    def __init__(self, root: str, seg:str,train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "images"
        self.ann = "ann"
        self.wt = "wt"
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(root+self.flag , seg)) if i.endswith(".jpg")]
        self.img_list = [os.path.join(root+self.flag , seg, i) for i in img_names]
        
        # check files
    

        self.roi_mask = [os.path.join(root+self.ann, seg, i[:-4]+'.png')
                         for i in img_names]
        self.wt_list = [os.path.join(root+self.wt, seg, i[:-4]+'.npy')for i in img_names]
        # check files
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.roi_mask[idx])
        mask_wt = torch.tensor(np.load(self.wt_list[idx]).astype(float))
        
        if torch.max(mask_wt) > 0:
            mask_wt = mask_wt / torch.max(mask_wt)
        
        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        #mask = Image.fromarray(np.array(mask.long()))

        if self.transforms is not None:
            img,mask= self.transforms(img,mask)

        return img, mask,mask_wt.reshape(320,320)

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets,wt = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        batched_wt = cat_list(wt, fill_value=255)
        return batched_imgs, batched_targets,batched_wt


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
