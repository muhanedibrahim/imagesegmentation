from ast import Return
from sched import scheduler
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
import cv2
torch.cuda.empty_cache()


pathofimages='2_Ortho_RGB/*tif'
pathoflabel='Potsdam/alllabel/*tif'
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
            if split(i)[2:4]==split(k)[4:6]:
                train_mask_id2.append(i)
                train_img_id2.append(k)
                
   
    val_img_id=[]
    val_mask_id=[]
    len_train_val=int(.2*len(train_img_id2))
    for i in range(len_train_val):
        id_label=random.randint(0,len(train_img_id2))
        val_img_id.append(train_img_id2[id_label])
        val_mask_id.append(train_mask_id2[id_label])
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
        image=cv2.imread(self.img_id[index])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        label=cv2.imread(self.mask_id[index])
        label=cv2.cvtColor(label,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(448,448))
        label=cv2.resize(label,(448,448))
        label=np.array(label)
        label2=np.zeros(shape=label.shape)
        for i in range(6):
          label2=np.where((np.array(label)==colors[i]).all(axis=-1,keepdims=True),i,label2)

        label2=np.expand_dims(label2[:,:,0],axis=-1)

        img=(torch.from_numpy(np.array(image))/255.0).type(torch.float32)
        mask=(torch.from_numpy(label2)).long()
        img=img.permute(2,0,1)
        mask=mask.permute(2,0,1)
       
        if self.preprocessing:
            if index in indexes:
                img, mask  = self.preprocessing(img, mask)
        
        mask=mask.squeeze(axis=0)
    
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

train_loader = DataLoader(alldata, batch_size=8,num_workers=8,shuffle=True)
val_loader = DataLoader(allval, batch_size=8,num_workers=1,shuffle=True)
train_histogram=[]
for i in alldata:
    train_histogram.append(np.array(i[1]))

train_histogram,x=np.histogram(train_histogram, bins=[i for i in range(7)], range=7)

print(train_histogram)
weight=torch.from_numpy(np.array([1-(x/sum(train_histogram)) for x in train_histogram])).float()
print(weight)

los = nn.CrossEntropyLoss(weight.to('cuda'))

model=seg.UnetPlusPlus(encoder_name='resnet18',in_channels=3,encoder_weights='imagenet',classes=6,activation=None).to('cuda')
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.001),
])
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, 
patience=6,verbose=True)

val_loss_min=100
def Train_one_epoch():
    train_loss=0
    for image, mask in train_loader:
        optimizer.zero_grad()
        image=image.to('cuda')
        mask=mask.to('cuda')
        res=model(image)
        loss=los(res,mask)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()*image.size(0)
    return train_loss/len(train_loader.sampler)
def val_one_epoch():
    val_loss=0
    for image, mask in val_loader:
        image=image.to('cuda')
        mask=mask.to('cuda')
        res=model(image)
        loss=los(res,mask)
        val_loss+= loss.item()*image.size(0)
    return res,val_loss/len(val_loader.sampler)

for i in range(200):
    train_loss=Train_one_epoch()
    res,val_loss=val_one_epoch()
    scheduler.step(val_loss)

    if val_loss<val_loss_min:
        val_loss_min=val_loss
        torch.save(model,'saved_model.pth')
        if val_loss_min<0.3:
            torchvision.utils.save_image(torch.argmax(res,dim=-1),'images/{}'.format(i+1))
    if (i+1)%5==0:
        print('epoch= {},  train_loss= {},  val_loss= {}'.forma(i+1,train_loss,val_loss))


