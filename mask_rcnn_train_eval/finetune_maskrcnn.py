 # -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:00:31 2022

For finetuning a trained Mask R-CNN model. We finetune with real images. Training settings can be set in settingsfile.
"""
import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
import torch.utils.data
import cv2 as cv
import torchvision.models.segmentation
import torch
import os
from torchvision.transforms import transforms as transforms
import matplotlib.patches as patches
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from matplotlib import pyplot as plt
import sys
import decimal
from load_json import *
import torchvision.transforms as TF

import pandas as pd
# import xlsxwriter
import matplotlib.patches as patches
from openpyxl import load_workbook

#%%
#np.set_printoptions(threshold=sys.maxsize)
imageSize = [400,400]
# random augmentation
transforms = TF.RandomApply(torch.nn.ModuleList([TF.ColorJitter(),]), p=0.3)
img_aug = torch.jit.script(transforms) 
def TF():
  true_false = random.choice([True,False])
  return true_false
  
def decnum(Lbound,Ubound):
    N = float(decimal.Decimal(random.randrange(Lbound*100, Ubound*100))/100) 
    print(N)
    return N

def load_real(fname):
    '''
    Returns
    -------
    img : loaded image
    masks : all masks, boolean

    '''
    # load json file
    path = r""  # location of json files
    f = open(path+fname)
    data = json.load(f)
    
    imagePath = data['imagePath']
    img = cv.imread(os.path.join(path+imagePath))        # load image
    
    polyg = data['shapes']      # obtain the polygon shapes
    num_obj = len(polyg)        # number of objects in image
    
    masks = []
    ## now convert polygons to binary masks
    for object in polyg:
        points = object['points']
        label = object['label']
        shape = img.shape
        mask = np.zeros(shape[0:2])
        cv.fillPoly(mask,np.int32([points]),1)
        mask = mask.astype(bool)
        mask = cv.resize(np.uint8(mask), imageSize, interpolation = cv.INTER_NEAREST)
        plt.figure()
        plt.imshow(mask)
        masks.append(mask)
       
    img = cv.resize(img, imageSize, interpolation =  cv.INTER_LINEAR)
    plt.figure()
    plt.imshow(img)  
    return img, masks



def loadData(ind):
    
    batch_Imgs = []
    batch_Data = []
    for idx in ind:

        if imnames[idx].endswith('.json'):
            # then real image, so create torch stuff from json file
            print('naam', imnames[idx])
            img, mask = load_real(imnames[idx])
            num_objs = len(mask) 
            msk = torch.zeros([num_objs,imageSize[1],imageSize[0]], dtype=torch.uint8)
            boxes = torch.zeros([num_objs,4], dtype=torch.float32)
            for m, i in zip(mask,range(len(mask))): # do for each object instance
                msk[i,:,:] = torch.tensor(m,dtype=torch.uint8)
    
                pos = np.where(m)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes[i] = torch.tensor([xmin, ymin, xmax, ymax])
            
            img = torch.as_tensor(img, dtype=torch.float32)
        
            # ALL DATA IN ONE DICT
            data = {}
            data["boxes"] = boxes
            Labels = torch.ones((num_objs,), dtype=torch.int64)   
            data["labels"] = torch.ones((num_objs,), dtype=torch.int64)   
            data["masks"] = msk#masks_torch.swapaxes(0,2)
            batch_Imgs.append(img)
            batch_Data.append(data) 
            
        else: 
            print(imnames[idx])
            print(masknames[idx])
            img = cv.imread(os.path.join(impath + imnames[idx]))
            mask = cv.imread(os.path.join(maskpath + masknames[idx]))
            
            img = cv.resize(img, imageSize, interpolation =  cv.INTER_LINEAR)
            mask = cv.resize(mask, imageSize, interpolation = cv.INTER_NEAREST)
    
            #% look for different values in mask images
            unique, counts = np.unique(mask, return_counts=True) #looks for all values 0 is background
                
            num_objs = len(unique)-1 # -1 because value 0 indicates background
            print('NUMOBJS',num_objs)
            boxes = torch.zeros([num_objs,4], dtype=torch.float32)
            msk = torch.zeros([num_objs,imageSize[1],imageSize[0]], dtype=torch.uint8)
            
            # print('image',n)
            for u, i in zip(unique[1:],np.arange(0,num_objs)): # do for each object instance
                obj_msk1 = np.where(mask == u, 1, 0)
                
                obj_msk = np.prod(obj_msk1, axis=-1)
                msk[i,:,:] = torch.tensor(obj_msk,dtype=torch.uint8)#.append(obj_msk)
    
                pos = np.where(obj_msk)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes[i] = torch.tensor([xmin, ymin, xmax, ymax])
            
            img = torch.as_tensor(img, dtype=torch.float32)
        
            # ALL DATA IN ONE DICT
            data = {}
            data["boxes"] = boxes
            Labels = torch.ones((num_objs,), dtype=torch.int64)   
            data["labels"] = torch.ones((num_objs,), dtype=torch.int64)   
            data["masks"] = msk#masks_torch.swapaxes(0,2)
            batch_Imgs.append(img)
            batch_Data.append(data) 
        
        # CONVERT IMAGE DATA TO PYTORCH FORMAT
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs],0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data



# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')


plt.close('all')
#%% get the images
trainDir="" # train data (for finetuning the json data) 
testDir=r"" 

folder = ""  # location of the saved trained model
tmodel = "60_v46.torch"
#

model=torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT') 


n_classes = 2 # background + 1 object
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=n_classes)


in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,256,n_classes)



model.load_state_dict(torch.load(folder + tmodel))  # load our model
# 

model.to(device)# move model to the right device
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-7,weight_decay=1e-9)
                                
print('training')
model.train() # train

#%%
    
lossvals = []
v = 46

savedir =""  # where to save the model

exname = "output_v" + str(v) + ".xlsx"

scaler = torch.cuda.amp.GradScaler(init_scale=8192.0)

imnames = []
impath = trainDir
for pth in os.listdir(impath):
    imnames.append(pth)

L = len(imnames)
batchSize = 2
# masknames.sort() # then we have both lists in the right order
imnames.sort()

for e in range(11):
    
    
    # for idx in range(len(imnames)): # range(1): #
    samplelist = np.array(range(0,L)) 
  
    rest = samplelist.size % batchSize # divide training set in 'rest' amount of batches
    
    while samplelist.size:
    
        idx = []
        if samplelist.size < batchSize:    # should only be the last batch if the training set size/batchSize is not an integer. This batch will be smaller than batchSize
            samples = range(0,rest)
        else:
            samples = range(0,batchSize)   # 'normal' batch sample numbers
        
        np.savetxt('samples_v'+str(v)+'.out', samples)
        
        for n in samples: # Select random instances in the training set. These indices will be deleted from the samplelist s.t. no repetition is present in one epoch.
            num = random.choice(samplelist)
            idx.append(num)
            samplelist = np.delete(samplelist,np.where(samplelist == num))

        images,targets = loadData(idx)
        newim = img_aug(images)
        
        images = list(image.to(device) for image in newim)  # contains all images
        targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
           
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
           
        
        # update nn with backpropagation
        if device.type == 'cpu':
            losses.backward()
            optimizer.step()
        else: 
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

        
           
        print(e,'loss:', losses.item())
        lossvals.append(losses.item())
        
        dframe = pd.DataFrame(lossvals)
        
    
        if e == 0:
            writer = pd.ExcelWriter(exname,engine='openpyxl')
            dframe.to_excel(writer)
            writer.save() 
        else:
            with pd.ExcelWriter(exname, engine = 'openpyxl',mode='a',if_sheet_exists='replace') as writer:
                writer.book = load_workbook(exname)
                writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
                reader = pd.read_excel(exname)
                dframe.to_excel(writer)
    
    # save every once in a while
    if e%1 == 0: # depends on how many epochs we do, make it larger for more epoch
            name = str(e)+"_v"+str(v)+"_ft.torch"
            torch.save(model.state_dict(), savedir + name)
            print("Save model to:"+ name)
               

