 # -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:00:31 2022

Evaluate Mask R-CNN by calculating the Average Precision (AP)
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
#from load_json import *
import pandas as pd
import json




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
                value = val[0].strip()
                dict_vars[varname] = value.replace('\n','')
    return dict_vars
    
    
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
    path = r"./chicken_dataset/jpg/"  # this folder should contain jpg
    f = open(path+fname)
    data = json.load(f)
    
    imagePath = data['imagePath']
    img = cv.imread(os.path.join(path+imagePath))        # load image
    #plt.figure()
    #plt.imshow(img)
    #plt.show()
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
        #plt.figure()
        #plt.imshow(mask)
        #plt.show()
        masks.append(mask)
    #plt.figure()
    #plt.imshow(img)  
    img = cv.resize(img, imageSize, interpolation =  cv.INTER_LINEAR)

    return img, masks



def loadData(idx):
    batch_Imgs = []
    batch_Data = []

    if imnames[idx].endswith('.json'):
        # then real image, so create torch stuff from json file
        #print('image name: ', imnames[idx])
        img, mask = load_real(imnames[idx])
        num_objs = len(mask) 
        msk = torch.zeros([num_objs,1,imageSize[1],imageSize[0]], dtype=torch.uint8)
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        for m, i in zip(mask,range(len(mask))): # do for each object instance
            # plt.figure()
            # plt.imshow(obj_msk1)
            msk[i,:,:,:] = torch.tensor(m,dtype=torch.uint8)#.append(obj_msk)

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
        msk = torch.zeros([num_objs,1,imageSize[1],imageSize[0]], dtype=torch.uint8)
        
        # print('image',n)
        for u, i in zip(unique[1:],np.arange(0,num_objs)): # do for each object instance
            obj_msk1 = np.where(mask == u, 1, 0)
            
            obj_msk = np.prod(obj_msk1, axis=-1)
            msk[i,:,:,:] = torch.tensor(obj_msk,dtype=torch.uint8)#.append(obj_msk)

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


settingsfile = r"./test_init.txt"
with open(settingsfile) as f:
    text = f.readlines()
    set_vars = makevar()
    
trainDir = set_vars['traindir']   
#testDir = set_vars['testdir']
testDir = r"./chicken_dataset/json/Test/" 

#folder = r"/home/Downloads/" 
#tmodel = set_vars['tmodel'] # to test all models, we just read all .torch files in the version folder

#tmodel = r"./MRCNN_chicken.torch"
tmodel = r"./MRCNN_chicken_finetuned.torch"

h = int(set_vars['imsize_h'])
w = int(set_vars['imsize_w'])

imageSize = [h,w]
print('imsize',imageSize)
# obtain all pngimages and masks
maskpath = testDir + 'Masks/'
impath = testDir + 'PNGimages/'

imnames = []
masknames = []

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')


plt.close('all')

#
model=torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT') 


n_classes = 2 # background + 1 object
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=n_classes)


in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,256,n_classes)


model.load_state_dict(torch.load(tmodel, map_location=torch.device('cpu')))#folder + tmodel))  # load our model

model.to(device)# move model to the right device
model.eval()


# initialize save arrays
bboxAP = []
bboxAP_50 = []
bboxAP_75 = []
bboxAP_S = []
bboxAP_M = []
bboxAP_L = []

maskAP = []
maskAP_50 = []
maskAP_75 = []
maskAP_S = []
maskAP_M = []
maskAP_L = []
mmaskAP = []
mmaskAP_50 = []
mmaskAP_75 = []
    

IoU = []


impath = testDir# + 'PNGimages/'


for pth in os.listdir(impath):
    imnames.append(pth)

# masknames.sort() # then we have both lists in the right order
imnames.sort()

# precision/recall
P = []
P_50 = []
P_75 = []
R = []
R_50 = []
R_75 = []

th1 = np.arange(0.5,1.0,0.05)
TP = np.zeros([1,len(th1)])
FP = np.zeros([1,len(th1)])
TP_50 = []
FP_50 = []
TP_75 = []
FP_75 = []

recall = []
precision = []
c=0     # represents cth prediction
scoredict = []
total_obj = 0
total_pred = 0

print(imnames)
for idx in range(len(imnames)): # range(1): #
    print('image ',idx, ' out of ', len(imnames))
    print('image name: ', imnames[idx])
    if imnames[idx].endswith('.json'):
        imname = imnames[idx].removesuffix('.json') + '.jpg'
        imgPath =  r"./chicken_dataset/json/Test/" + imname
    else: 
        imname = imnames[idx]   
        imgPath = impath + imname  # not in trainset
    
    
    
    ## load target
    tar_img, tar_data = loadData(idx)#imgPath,maskPath)


    ### load testimg
    print(imgPath)
    imPath = r"./chicken_dataset/jpg/" + imname
    img = cv.imread(imPath)
    # print(img)
    img = cv.resize(img, imageSize, interpolation = cv.INTER_LINEAR)
    
    
    imgrgb = img
    
    images = torch.as_tensor(imgrgb, dtype = torch.float32).unsqueeze(0)
    images = images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(device) for image in images)

    im = imgrgb.copy()
    with torch.no_grad():
        pred = model(images)
   
    
    # plt.figure()
    # plt.imshow(im)
    
    #cv.imwrite('gt_image.jpg',im)
    
    B = pred[0]['boxes'].tolist()
    
    score_threshold = 0#.999 #AP is calculated for ALL predictions
    scores = pred[0]['scores'][:].detach().cpu().numpy()
    scoresth = np.where(scores > score_threshold)[0]
    

    print('scores: ', scores)
    print(str(len(scoresth)) + ' masks with scr>' + str(score_threshold) + ' predicted. There are ' 
    + str(len(tar_data[0]['masks'])) + ' objects in gt. \n' + 'In total ' + str(len(B)) + ' masks predicted.' )
  
    # we have the predicted masks, we have the target masks: pred & tar_data
    th1 = np.arange(0.5,1.0,0.05)
    th2 = 0.5
    th3 = 0.75            

    total_pred += len(scoresth)

    indsave = []
    for i in range(len(tar_data[0]['masks'])):
        if len(scoresth) < 1:
            total_obj+=1
        else:
            total_obj+=1
              # print('Target mask: ',i)
            target_mask = tar_data[0]['masks'][i,0].squeeze().detach().cpu().numpy()
            
            # cv.imwrite('gt_target.jpg',target_mask*255)
            
            # plt.figure(), plt.imshow(target_mask,cmap='gray')
            IoU = []
            scoresave = []
    
            for n in range(len(pred[0]['masks'])):
                scr=pred[0]['scores'][n].detach().cpu().numpy()
                # print('score', scr)
                if scr > score_threshold: # should be 0.99 or smth because only those would be 'good' masks
                    scoresave.append(scr)
                    msk=pred[0]['masks'][n,0].squeeze().detach().cpu().numpy() > 0.5
                    # print(msk.dtype)
                    
                    #cv.imwrite('prediction'+str(n)+'.jpg',msk*255)
                    
                    gt = target_mask > 0 # ground truth bool
                    # plt.figure('GT'), plt.imshow(gt,cmap= 'gray')
                    # plt.figure('PRED'),plt.imshow(msk,cmap = 'gray')
                    I = gt * msk # intersection
                    U = gt + msk # union
                    # plt.figure('UNION'),plt.imshow(U,cmap='gray')
                    # plt.figure('INTERSECTION'),plt.imshow(I,cmap='gray')
                   
                   	
                    #cv.imwrite('union'+str(n)+'.jpg',U*255)
                    #cv.imwrite('intersection'+ str(n)+'.jpg',I*255)
                    
                    # plt.show()
                    IoU.append(np.sum(I)/np.sum(U))
                    
                    print('IOU',IoU[-1])
            indx = np.argmax(IoU)
            indsave.append(indx)  # save selected instances
            maxIoU = IoU[indx]
            scoredict.append(scoresave[indx])

            
            for th, k in zip(th1, range(len(th1))):
                TP[c,k]+= maxIoU > th
                FP[c,k]+= maxIoU <= th
            c+=1
    
            if (i == len(tar_data[0]['masks'])-1) and (len(scoresth)>len(tar_data[0]['masks'])):
              
                indices = range(len(scoresth))
                for b in indsave:
                    if b in indices: # if index is NOT in highest IoUs, then automatic FP
                        indices = np.delete(indices,np.where(indices==b)[0])
               
                for b in indices:
                    scoredict.append(scoresave[b])
                    print(scoresave[b])
                    TP = np.concatenate((TP,np.zeros([1,len(th1)])))
                    FP = np.concatenate((FP,np.ones([1,len(th1)]))) 
                    c += 1

                  
            TP = np.concatenate((TP,np.zeros([1,len(th1)])))
            FP = np.concatenate((FP,np.zeros([1,len(th1)])))
        
#%%
TP_f = np.delete(TP,-1,0)
FP_f = np.delete(FP,-1,0)
# put it all in one df 
d = {'conf':scoredict}

for i in range(len(th1)):
     d['TP'+str(round(th1[i]*100))]=TP_f[:,i]
     d['FP'+str(round(th1[i]*100))]=FP_f[:,i]   

df = pd.DataFrame(data=d)
# sort by confidence score
#%%
df_sort = df.sort_values('conf',ascending=False)


#%%
for i in range(len(th1)):
     df_sort['acc_tp'+ str(round(th1[i]*100))] = np.cumsum(df_sort['TP'+str(round(th1[i]*100))])
     df_sort['acc_fp'+ str(round(th1[i]*100))] = np.cumsum(df_sort['FP'+str(round(th1[i]*100))])
     
# need cumulative FP and TP
for i in range(len(th1)):
     df_sort['precision'+ str(round(th1[i]*100))] = df_sort['acc_tp'+ str(round(th1[i]*100))]/(df_sort['acc_tp'+ str(round(th1[i]*100))]+df_sort['acc_fp'+ str(round(th1[i]*100))])
     df_sort['recall'+ str(round(th1[i]*100))] = df_sort['acc_tp'+ str(round(th1[i]*100))]/total_obj

plt.plot(df_sort['recall50'],df_sort['precision50'])

#%% interpolate

step = 0.1
n_points = (1+step)/step
R = np.arange(0,1 + step,step)#.1/n_points) ## 11 points, as we want AP11

savefolder = './results/'

if not os.path.exists(savefolder):
    os.makedirs(savefolder)

if 'ft' in tmodel:
    savename = 'ft_PR-curve_'
else:
    savename = 'synth_PR-curve_'
    
for i in range(len(th1)):
    P_interp = []
    # read recall and precision
    P = np.array(df_sort['precision'+ str(round(th1[i]*100))].tolist())
    recall = np.array(df_sort['recall'+ str(round(th1[i]*100))].tolist())
    for r in R:
        R_tilde = np.where(recall>=r)
        # print('R',R_tilde[0])
        if len(R_tilde[0])<1:
            P_interp.append(0)
        else:
            P_interp.append(np.amax(P[R_tilde]))
        # print(P_interp)
    fig = plt.figure(str(i))
    plt.plot(recall,P, label = "precision",linewidth=3)
    plt.plot(R, P_interp, label = "interp. precision", linestyle="", marker="o", markersize= 12)
    plt.xlabel('Recall',fontsize=30)
    plt.ylabel('Precision',fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.title('IoU: '+ str(round(th1[i]*100)),fontsize=30)
    plt.legend(fontsize=20)
    
    #ax = plt.gca()
    #ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    #plt.draw()
    #plt.savefig(savefolder + savename+'IoU-'+str(round(th1[i]*100))+'.png', bbox_inches='tight')
    fig.savefig(savefolder + savename+'IoU-'+str(round(th1[i]*100))+'.png', bbox_inches='tight')
    df_sort['AP11_'+str(round(th1[i]*100))] = np.sum(P_interp)/n_points
    
#%%
df_sort['APmean'] = df_sort['AP11_50']
for c in range(len(th1)-1):
    i = c+1
    df_sort['APmean'] += df_sort['AP11_'+str(round(th1[i]*100))] 
    
df_sort['APmean']=df_sort['APmean']/len(th1)


#%% save as xlsx
# ename = savefolder + tmodel[3:11] + "_mAP.xlsx"
# df_sort.to_excel(ename)  
# with pd.ExcelWriter(ename) as writer:  
#     df_sort.to_excel(writer, sheet_name='mAP_real')
