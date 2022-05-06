from ast import Return
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import torchvision.transforms.functional as TF
import torchvision
import matplotlib.pyplot as plt
import glob
import segmentation_models_pytorch as seg 
from torch import nn as nn
import random
device=torch.device('cuda')
import glob
import zipfile
import os

torch.cuda.empty_cache()


pathoflabel='2_Ortho_RGB/*tif'
pathofimages='Potsdam/alllabel/*tif'
indexes=[i for i in range(6)]
colors=[[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0]]

def return_all_items(data,all_items):
  
  for i in range(data.data_len):
    all_items.append(data[i])
  return all_items

def split(name):
  part=name.split('_')
  return part

#print(split(glob.glob(pathoflabel)[0])[4:6])
def get_train_id(train_path,mask_path):
    train_img_id=glob.glob(train_path)
    train_mask_id=glob.glob(mask_path)
    train_mask_id2=[]
    train_img_id2=[]
    for i in (train_mask_id):
        for k in (train_img_id):
            if split(i)[4:6]==split(k)[2:4]:
                train_mask_id2.append(i)
                train_img_id2.append(k)
                
   
    val_img_id=[]
    val_mask_id=[]
    len_train_val=int(.2*len(train_img_id2))
    for i in range(len_train_val):
        id_label=random.randint(0,len(train_img_id2))
        val_img_id.append(train_img_id[id_label])
        val_mask_id.append(train_mask_id[id_label])
        train_img_id2.pop(id_label)
        train_mask_id2.pop(id_label)

    return val_img_id,val_mask_id, train_img_id2,train_mask_id2

val_img_id,val_mask_id,train_img_id,train_mask_id=get_train_id(pathofimages,pathoflabel)

class dataset(BaseDataset):
    def __init__(self,img_id,mask_id,indexes,colors,preprocessing=None,datapass=None) :
            self.img_id=img_id
            self.mask_id=mask_id
            self.preprocessing=preprocessing
            self.data_len=len(self.img_id)
            self.indexs=indexes
            self.colors=colors
            self.datapass=datapass
    
    def __getitem__(self, index):
        image=Image.open(self.img_id[index])
        label=Image.open(self.mask_id[index])
        image=image.resize((448,448))
        label=label.resize((448,448))
        label=np.array(label)
        label2=np.zeros(shape=label.shape)
        for i in range(6):
          label2=np.where((np.array(label)==colors[i]).all(axis=-1,keepdims=True),i,label2)

        label2=np.expand_dims(label2[:,:,0],axis=-1)

        img=(torch.from_numpy(np.array(image))/255.0).type(torch.float32)
        mask=(torch.from_numpy(label2)).type(torch.LongTensor)
        img=img.permute(2,0,1)
        mask=mask.permute(2,0,1)
       
        if self.preprocessing:
            if index in indexes:
                img, mask  = self.preprocessing(img, mask)
        
        #mask=mask.squeeze(axis=0)
    
        return img,mask

    def __len__(self):
        return self.data_len
class augmen():
    def __call__(self,img,mask):
        return torchvision.transforms.ColorJitter(brightness=(0.1,1.3), contrast=((0.1,1.3)), saturation=(0.1,1.3), hue=(0,.5))(img),mask
class augmen4():
    def __call__(self,img,mask):
        point=random.randint(5,30)
        return TF.resized_crop(img,top=point,left=point,height=300,width=300,size=[448,448]),TF.resized_crop(mask,top=point,left=point,height=300,width=300,size=[448,448])
class augmen2():
    def __call__(self,img,mask): 
        angel=round(random.uniform(10.33, 20.66), 2)
        return TF.rotate(img,angel),TF.rotate(mask,angel)


indexes=[random.randint(0,30) for i in range(15)]
data=dataset(train_img_id,train_mask_id,indexes,colors,None)
augmenteddata=dataset(train_img_id,train_mask_id,indexes,colors,augmen())
augmenteddata2=dataset(train_img_id,train_mask_id,indexes,colors,augmen2())
augmenteddata3=dataset(train_img_id,train_mask_id,indexes,colors,augmen4())
val=dataset(val_img_id,val_mask_id,indexes,colors,None)
augmentedval=dataset(val_img_id,val_mask_id,indexes,colors,augmen())
augmentedval2=dataset(val_img_id,val_mask_id,indexes,colors,augmen2())
augmentedval3=dataset(val_img_id,val_mask_id,indexes,colors,augmen4())



all_items=[]
alldata=return_all_items(data,all_items)
alldata=return_all_items(augmenteddata,alldata)
alldata=return_all_items(augmenteddata2,alldata)
alldata=return_all_items(augmenteddata3,alldata)

all_items=[]
allval=return_all_items(val,all_items)
allval=return_all_items(augmentedval,allval)
allval=return_all_items(augmentedval2,allval)
allval=return_all_items(augmentedval3,allval)
print(len(alldata),len(allval))

train_loader = DataLoader(alldata, batch_size=8,num_workers=8,shuffle=True)
val_loader = DataLoader(allval, batch_size=8,num_workers=1,shuffle=True)
imgtest,labeltest=next(iter(train_loader))
print(imgtest.shape,labeltest.shape)

print(imgtest.shape,labeltest.shape)

loss = seg.utils.losses.DiceLoss()

matrices=[seg.utils.metrics.IoU(threshold=0.5)]

model=seg.UnetPlusPlus(encoder_name='resnet18',in_channels=3,encoder_weights='imagenet',classes=6,activation=None).to('cuda')

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.001),
])
train_epoch = seg.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=matrices, 
    optimizer=optimizer,
    device='cuda',
  
    verbose=True,
)

valid_epoch = seg.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=matrices, 
    device='cuda',
    
    verbose=True,
)



max_score=0
for i in range(100):
  print('epoch {}'.format(i))
  train_res=train_epoch.run(train_loader)
  val_res=valid_epoch.run(val_loader)
  if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-4
  if i == 80:
        optimizer.param_groups[0]['lr'] = 1e-8
  
  if max_score<val_res['iou_score']:
      max_score=val_res['iou_score']
      torch.save(model, 'best_model2.pth')


