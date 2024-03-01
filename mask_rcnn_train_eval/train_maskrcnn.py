# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 14:44:52 2022

Train Mask R-CNN using pytorch. Set the training details in the settings.txt file.
"""

import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
import torch.utils.data
import cv2 as cv
import torchvision.models.segmentation
import torch
import time
import os

import torchvision.transforms as TF


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')


# random augmentation
transforms = TF.RandomApply(torch.nn.ModuleList([
    TF.ColorJitter(),
]), p=0.3)
img_aug = torch.jit.script(transforms) # only use this on the images because we don't want to altr the mask values




def makevar(): # make dictionary including all variables from txt
    dict_vars = {}
    for line in text:                   # read all lines
#        print(line)
        if line[0] != "#":
            if "=" in line:             # then make var
                arr = line.split("=")   # after : is the value
                varname = arr[0]
                val = arr[1]
                if "#" in val:          # don't look at comment
                    val = val.replace('\t','')
                val = val.split("#")
#                print('RRRRR',val)
                value = val[0].strip()
                dict_vars[varname] = value.replace('\n','')
    return dict_vars

settingsfile = r"./train_init.txt"
with open(settingsfile) as f:
    text = f.readlines()
    set_vars = makevar()
    
    
trainDir = set_vars['traindir']
testDir = set_vars['testdir']

v = set_vars['modelnum']#'35'

l_rate = float(set_vars['learning_rate'])#1e-5
w_decay = float(set_vars['weight_decay'])# 1e-8

h = int(set_vars['imsize_h'])
w = int(set_vars['imsize_w'])

batchSize = int(set_vars['batchsize'])# 4#10
imageSize = [h,w]

epochs = int(set_vars['epochs'])

save_session = set_vars['save_all_in'] + 'v' + v + '/' # to save all relevant files for later backup

pretrained_weights = int(set_vars['pretrained'])

exists = os.path.exists(save_session)
if not exists:
    os.makedirs(save_session)




imnames = []
masknames = []

maskpath = trainDir + 'Masks/'
impath = trainDir + 'PNGimages/'

for pth in os.listdir(maskpath):
    masknames.append(pth)

for pth in os.listdir(impath):
    imnames.append(pth)
    
masknames.sort() 
imnames.sort()

#for pth in os.listdir(trainDir):
 #   if pth[0:4] == 'Mask':
  #      masks.append(trainDir+"/"+pth)
   # else:
    #    imgs.append(trainDir+"/"+pth)




batch_Imgs=[]
batch_Data=[]

#imageSize=[random.randint(500,800),random.randint(500,800)]

# https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
#%%
#%%
L = len(imnames)
bS = range(0,L)



def TF():
  true_false = random.choice([True,False])
  return true_false
  
#%%
def loadData(idx):
    #imageSize=[400,400]#[random.randint(500,800),random.randint(500,800)] # set random size for images, changes every epoch
    batch_Imgs = []
    batch_Data = []
    for ind in idx:
        print(imnames[ind])
        #print('INDEX',ind)
        img = cv.imread(os.path.join(impath+imnames[ind]))
        mask = cv.imread(os.path.join(maskpath+masknames[ind]))
                  
        img = cv.resize(img, imageSize, interpolation =  cv.INTER_LINEAR)
        mask = cv.resize(mask, imageSize, interpolation =  cv.INTER_NEAREST)
        
    #% look for different values in mask images
        unique, counts = np.unique(mask, return_counts=True) #looks for all values 0 is background
        #print('imgname',imnames[ind])
        print('UNIQUE',unique)
        

    # create binary masks for all imgs
        inst = []
  
        num_objs = len(unique)-1 # -1 because value 0 indicates background
        #print('NUMOBJS',num_objs)
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        msk = torch.zeros([num_objs,imageSize[1],imageSize[0]], dtype = torch.uint8)

            #print('image',imnum)
        check = 0
           
        for u,i in zip(unique[1:],range(num_objs)): # do for each object instance
        
        # print('obj',num_objs)    
        # print('unique vals',unique)
            print('u',u)
        # print('i',i)
            obj_msk = np.where(mask == u, 1, 0)
            obj_msk = np.prod(obj_msk, axis=-1)

            pos = np.where(obj_msk)

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
                
            print([xmin, ymin, xmax, ymax])        
       
            boxes[i] = torch.tensor([xmin, ymin, xmax, ymax])
            msk[i,:,:] = torch.tensor(obj_msk,dtype = torch.uint8)
             
        img = torch.as_tensor(img, dtype=torch.float32)
 
        data = {}
        data["boxes"] = boxes
        data["labels"] = torch.ones((num_objs,), dtype=torch.int64)   
        #masks_swap = msk.swapaxes(0,2)#masks_torch.swapaxes(0,2)
        data["masks"] = msk#masks_swap
        

        batch_Imgs.append(img)
        batch_Data.append(data)
            
        # CONVERT IMAGE DATA TO PYTORCH FORMAT
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs],0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data


if pretrained_weights:
    wts = 'MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT'
    model=torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=wts)  # choose pretrained weights
else: 
    model=torchvision.models.detection.maskrcnn_resnet50_fpn()
    
    
n_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes = n_classes)
# change num_classes according to how many different types of objects there are
# for the test, only chicken fillets, so num_classes = 1 + background

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,256,n_classes)

model.to(device) # load model to gpu/cpu

optimizer = torch.optim.AdamW(params=model.parameters(), lr=l_rate, weight_decay=w_decay) # optimizer?

#print('training')
model.train() # train

#exname = saveDir + "output_v" + str(v) + ".xlsx"

lossvals = []

n_epochs = epochs

### print important information for later reference
print('### TRAIN NETWORK WITH SYNTHETIC DATA ###')
print('Python script:   ', str(os.path.basename(__file__)))
print('Model version:   ', v)
print('Train directory: ', trainDir)
print('Number of imgs:  ', L)
print('Batchsize:       ', batchSize)
print('Learning rate:   ', l_rate)
print('Weight decay:    ', w_decay)
print('Number of epochs:', n_epochs)

np.savetxt(save_session + 'info.txt', 
                      ['Python script:     ' + str(os.path.basename(__file__)),
                      'Model version:     ' + v , 
                      'Train directory:   ' + trainDir ,
                      'Number of imgs:    ' + str(L), 
                      'Batchsize:         ' + str(batchSize), 
                      'Learning rate:     ' + str(l_rate),  
                      'Weight decay:      ' + str(w_decay), 
                      'Number of epochs:  ' + str(n_epochs), 
                      'Imagesize:         ' + str(imageSize),
                      'Pretrained:        ' + str(pretrained_weights),
                      'Weights:           ' + wts], 
                      fmt='%s')


tic = time.perf_counter()

scaler = torch.cuda.amp.GradScaler(init_scale=8192.0)


for e in range(0,n_epochs+1): # epoch
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
            
        print('INDICES',idx)
        # load data
        images,targets = loadData(idx)
        newim = img_aug(images)
       
        images = list(image.to(device) for image in newim)  # contains all images
                 
        targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
           
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            
            
        losses = sum(loss for loss in loss_dict.values())
        lossvals.append([losses.item(),e])
        np.savetxt(save_session + 'losses_v'+str(v)+'.out', lossvals, delimiter=',')
        # update nn with backpropagation
        scaler.scale(losses).backward()
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        
        
       
    #print(e,'loss:', losses.item())
    
    # save every 50th torch file 
    if e%20== 0: # depends on how many epochs we do, make it larger for more epochs
            torch.save(model.state_dict(), save_session + str(e)+"_v"+ str(v)+".torch")
            print("Save model to:",str(e)+ "_v"+ str(v)+".torch")
  
## record training time: stop
toc = time.perf_counter()
print(f"Trained the network in {toc - tic:0.4f} seconds") 

