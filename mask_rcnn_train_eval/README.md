# Scripts for training and finetuning Mask R-CNN

## train_maskrcnn.py
Script for training Mask R-CNN. Use `train_init.txt` to set the right directories and desired training
values

## finetune
Script for finetuning the model trained with `train_maskrcnn.py`. Names and values have to be set in this file. 

## mAP_outputsxlsx.py
Test the trained models (indicate which one), calculate the mAP for multiple IoU thresholds and save 
the plots as images. Also output an .xlsx file for each IoU threshold
- TP
- FP
- accumulated TP 
- accumulated FP
- precision
- recall
- average precision (AP)
- average AP for IoU 50:5:95 

To use this file make sure the .torch files and images are in the right directories.
