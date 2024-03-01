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

from matplotlib import pyplot as plt

def load_model(tmodel,n_classes):
    print('Loading network...')
    device = torch.device('cpu')#torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model=torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT') 

    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=n_classes)


    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,256,n_classes)

    model.load_state_dict(torch.load(tmodel,map_location=torch.device('cpu')))  # load our model
    model.to(device)# move model to the right device
    model.eval()
    return model
    

def do_segmentation(img,model):
    print('Instance segmentation')
    
    imsize_init = img.shape
    device = torch.device('cpu' )#torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # resize image
    imageSize = (400,400)
    img = cv.resize(img, imageSize, interpolation = cv.INTER_LINEAR)
    imgrgb = img
    
    images = torch.as_tensor(imgrgb,dtype=torch.float32).unsqueeze(0)
    images = images.swapaxes(1,3).swapaxes(2,3)
    images = list(image.to(device) for image in images)

    im = imgrgb.copy()
    with torch.no_grad():
        pred = model(images) 

    B = pred[0]['boxes'].tolist()
    scores = pred[0]['scores'].detach().cpu().numpy()
    msks = []

    if not B:
        print('Nothing found...') 
    else:
        
        for i in range(len(pred[0]['masks'])):
            msk=pred[0]['masks'][i,0].squeeze().detach().cpu().numpy()

            scr=pred[0]['scores'][i].detach().cpu().numpy()
            print(scr)

            
            if scr>0.9999 : 

                rec = B[i]
                xy = rec[0:2]           # x0,y0
                width = rec[2]-rec[0]   # x1-x0
                height = rec[3]-rec[1]  # y1-y0
                rect = patches.Rectangle(xy,width,height,linewidth=1, edgecolor='r', facecolor='none')

                msks.append(cv.resize(msk,(imsize_init[1],imsize_init[0]),interpolation=cv.INTER_LINEAR))
                
    return msks, B, scores

