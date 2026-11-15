"""
fundus preprocessing, extracting the ROI, cropping and making square
source:     EyeQ/EyeQ_preprocess/fundus_prep.py
"""
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
import argparse


def get_args_train():
    parser = argparse.ArgumentParser(description="Image Classification Training")
    parser.add_argument("--experiment_name", type=str, default="experiment_default",
                        help="name of the experiment for saving models")    
    parser.add_argument("--csv_path", type=str, default="./data.csv",
                        help="Path to CSV file with image paths and labels")
    parser.add_argument("--model_name", type=str, default="ViT",
                        choices=["ViT", "convnext","RETFoundGreen"],
                        help="Model architecture to use")
    parser.add_argument("--train_strategy", type=str, default="none",
                        choices=["none", "deepensemble","evidential"],
                        help="train strategy suitable for target uncertainty method")
    parser.add_argument("--num_ensemble_models", type=int, default=3,
                        help="Number of models in ensemble learning")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of output classes")   
    parser.add_argument("--image_size", type=int, default=392,
                        help="Input image size (H and W)")
    parser.add_argument("--batch_size", type=int, default=3,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate")
    parser.add_argument("--patience_epochs_LRplateau", type=int, default=10,
                        help="Patience epochs for LR scheduler")
    parser.add_argument("--patience_epochs_early_stop", type=int, default=15,
                        help="Patience epochs for early stopping")        
    
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Loss function: cross-entropy (ce) or focal loss (focal)."
    )

    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=3.0,
        help="Base gamma for focal loss when using fixed schedule."
    )

    parser.add_argument(
        "--focal_schedule",
        type=str,
        default="fixed",
        choices=["fixed", "flsd53"],
        help="Focal loss schedule: 'fixed' (gamma constant) or 'flsd53' (sample-dependent 5/3).",
    )


    return parser.parse_args()


def get_args_inference():
    parser = argparse.ArgumentParser(description="Image Classification Inference")
    # parser.add_argument("--experiment_name", type=str, default="experiment_default",
    #                     help="name of the experiment for saving models")
    parser.add_argument("--csv_path", type=str, default="./data.csv",
                        help="Path to CSV file with image paths and labels")
    parser.add_argument("--experiment_path", type=str, default="./trainedModels/experiment_default",
                           help="path to the saved experiment including the trained model file")
    parser.add_argument("--UQ_method", type=str, default="None", nargs='+',
                        choices=['McDropout', 'TTA', 'None','DeepEnsemble', 'MSP',
                                 'PredictiveEntropy', 'PredictiveEntropy_tsallis', 
                                 'MSP_tuned', 'PredictiveEntropy_tuned', 'PredictiveEntropy_tsallis_tuned',
                                 'Conformal'],
                        help="uncertainty quantification strategy")
    parser.add_argument("--MC_DROPOUT_SAMPLES", type=int, default=50,
                        help="Number of forward passes for MC Dropout")
    parser.add_argument("--TTA_SAMPLES", type=int, default=20,
                        help="Number of augmented samples for TTA")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of output classes")
    parser.add_argument("--image_size", type=int, default=392,
                        help="Input image size (H and W)")
    return parser.parse_args()
    

def write_image(file_path, image):
    # if image.ndim == 3 and image.shape[2] == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, image)

def get_mask_BZ(img):
    if img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    threhold = np.mean(gray_img)/3-5 
    #cv2.imshow('gray_img', gray_img)
    #cv2.waitKey()
    #print(threhold)
    _, mask = cv2.threshold(gray_img, max(5,threhold), 1, cv2.THRESH_BINARY)
    
    #cv2.imshow('bz_mask', mask*255)
    #cv2.waitKey()
    nn_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8)
    new_mask = (1-mask).astype(np.uint8)
    #  cv::floodFill(Temp, Point(0, 0), Scalar(255));
    # _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, [(0, 0),(0,new_mask.shape[0])], (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (0,0), (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1]-1,new_mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,  20))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask


def _get_center_by_edge(mask):
    center=[0,0]
    x=mask.sum(axis=1)
    center[0]=np.where(x>x.max()*0.95)[0].mean()
    x=mask.sum(axis=0)
    center[1]=np.where(x>x.max()*0.95)[0].mean()
    return center


def _get_radius_by_mask_center(mask,center):
    mask=mask.astype(np.uint8)
    ksize=max(mask.shape[1]//400*2+1,3)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
    mask=cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    # radius=
    index=np.where(mask>0)
    d_int=np.sqrt((index[0]-center[0])**2+(index[1]-center[1])**2)
    b_count=np.bincount(np.ceil(d_int).astype(int))
    radius=np.where(b_count>b_count.max()*0.995)[0].max()
    return radius


def _get_circle_by_center_bbox(shape,center,bbox,radius):
    center_mask=np.zeros(shape=shape).astype('uint8')
    tmp_mask=np.zeros(shape=bbox[2:4])
    center_tmp=(int(center[0]),int(center[1]))
    center_mask=cv2.circle(center_mask,center_tmp[::-1],int(radius),(1),-1)
    # center_mask[bbox[0]:bbox[0]+bbox[2],bbox[1]:bbox[1]+bbox[3]]=tmp_mask
    # center_mask[bbox[0]:min(bbox[0]+bbox[2],center_mask.shape[0]),bbox[1]:min(bbox[1]+bbox[3],center_mask.shape[1])]=tmp_mask
    return center_mask


def get_mask(img):
    if img.ndim ==3:
        #raise 'image dim is not 3'
        g_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        #cv2.imshow('ImageWindow', g_img)
        #cv2.waitKey()
    elif img.ndim == 2:
        g_img =img.copy()
    else:
        raise 'image dim is not 1 or 3'
    h,w = g_img.shape
    shape=g_img.shape[0:2]
    #g_img = cv2.resize(g_img,(0,0),fx = 0.5,fy = 0.5)
    tg_img=cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
    tmp_mask=get_mask_BZ(tg_img)
    center=_get_center_by_edge(tmp_mask)
    #bbox=_get_bbox_by_mask(tmp_mask)
    #print(center)
    #cv2.imshow('ImageWindow', tmp_mask*255)
    #cv2.waitKey()
    radius=_get_radius_by_mask_center(tmp_mask,center)
    #resize back
    #center = [center[0]*2,center[1]*2]
    #radius = int(radius*2)
    
    center = [center[0], center[1]]
    radius = int(radius)
    s_h = max(0,int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h-s_h,2 * radius), min(w-s_w,2 * radius))
    tmp_mask=_get_circle_by_center_bbox(shape,center,bbox,radius)
    return tmp_mask,bbox,center,radius


def mask_image(img,mask):
    img[mask<=0,...]=0
    return img


def remove_back_area(img,bbox=None,border=None):
    image=img
    if border is None:
        border=np.array((bbox[0],bbox[0]+bbox[2],bbox[1],bbox[1]+bbox[3],img.shape[0],img.shape[1]),dtype=int)
    image=image[border[0]:border[1],border[2]:border[3],...]
    return image,border


def supplemental_black_area(img,border=None):
    image=img
    if border is None:
        h,v=img.shape[0:2]
        max_l=max(h,v)
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
        border=(int(max_l/2-h/2),int(max_l/2-h/2)+h,int(max_l/2-v/2),int(max_l/2-v/2)+v,max_l)
    else:
        max_l=border[4]
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
    image[border[0]:border[1],border[2]:border[3],...]=img
    return image,border


#def process_without_gb(img, label,radius_list,centre_list_w, centre_list_h):
def process_without_gb(img):
    # preprocess images
    #   img : origin image
    #   tar_height: height of tar image
    # return:
    #   result_img: preprocessed image
    #   borders: remove border, supplement mask
    #   mask: mask for preprocessed image
    borders = []    
    #print('image size before processing: ', img.shape)
    mask, bbox, center, radius = get_mask(img)
    #print('center is: ',center)
    #print('radius is: ',radius)
    r_img = mask_image(img, mask)
    r_img, r_border = remove_back_area(r_img,bbox=bbox)
    mask, _ = remove_back_area(mask,border=r_border)
    
    borders.append(r_border)
    r_img,sup_border = supplemental_black_area(r_img)
    #print('image size after processing: ', r_img.shape)
    
    mask,_ = supplemental_black_area(mask,border=sup_border)
    borders.append(sup_border)


    return r_img,borders,(mask*255).astype(np.uint8)

import os 
import random
class CustomPreprocessing:
    def __call__(self, img):
        # img is a PIL image at this point
        
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        r_img, borders, mask = process_without_gb(img)          

        # print('image size before processing: ', img.shape)
        # random_number = random.randint(1000, 9999)  # random 4-digit number
        # dst_path = os.path.join('./Dev/', f'processed_image_{random_number}.png')        
        # write_image(dst_path, r_img)
        # print('image size after processing: ', r_img.shape)

        img_pil = Image.fromarray(r_img)
        #print("after standard transform, image size:", img_pil.size)
        return img_pil



