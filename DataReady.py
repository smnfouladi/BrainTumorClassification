import cv2
from math import sin, cos, radians
import numpy as np
import glob
import os



glioma1 = "./Data/Training/glioma_tumor"
glioma2 = "./Data/Testing/glioma_tumor"

meningioma1="./Data/Training/meningioma_tumor"
meningioma2="./Data/Testing/meningioma_tumor"

pituitary1="./Data/Training/pituitary_tumor"
pituitary2="./Data/Testing/pituitary_tumor"

no_tumor1="./Data/Training/no_tumor"
no_tumor2="./Data/Testing/no_tumor"


for img_filename_g in os.listdir(glioma1):
    fname_g=img_filename_g.split(".jpg",1)[0]    
    modifiedname_g="glioma_"+fname_g    
    img1_g='./Data/Training/glioma_tumor/'+fname_g+'.jpg'
    img_g=cv2.imread(img1_g)        
    cv2.imwrite('./gliomaData/'+modifiedname_g+'.jpg', img_g)

for img_filename_g2 in os.listdir(glioma2):
    fname_g2=img_filename_g2.split(".jpg",1)[0]    
    img_filename_g2="glioma_"+fname_g2    
    img1_g2='./Data/Testing/glioma_tumor/'+fname_g2+'.jpg'
    img_g2=cv2.imread(img1_g2)        
    cv2.imwrite('./gliomaData/'+img_filename_g2+'.jpg', img_g2)

for img_filename_m in os.listdir(meningioma1):
    fname_m=img_filename_m.split(".jpg",1)[0]    
    modifiedname_m="meningioma_"+fname_m    
    img1_m='./Data/Training/meningioma_tumor/'+fname_m+'.jpg'
    img_m=cv2.imread(img1_m)        
    cv2.imwrite('./meningiomaData/'+modifiedname_m+'.jpg', img_m)

for img_filename_m2 in os.listdir(meningioma2):
    fname_m2=img_filename_m2.split(".jpg",1)[0]    
    modifiedname_m2="meningioma_"+fname_m2    
    img1_m2='./Data/Testing/meningioma_tumor/'+fname_m2+'.jpg'
    img_m2=cv2.imread(img1_m2)        
    cv2.imwrite('./meningiomaData/'+modifiedname_m2+'.jpg', img_m2)

for img_filename_p in os.listdir(pituitary1):
    fname_p=img_filename_p.split(".jpg",1)[0]    
    modifiedname_p="pituitary_"+fname_p    
    img1_p='./Data/Training/pituitary_tumor/'+fname_p+'.jpg'
    img_p=cv2.imread(img1_p)        
    cv2.imwrite('./pituitaryData/'+modifiedname_p+'.jpg', img_p)

for img_filename_p2 in os.listdir(pituitary2):
    fname_p2=img_filename_p2.split(".jpg",1)[0]    
    modifiedname_p2="pituitary_"+fname_p2    
    img1_p2='./Data/Testing/pituitary_tumor/'+fname_p2+'.jpg'
    img_p2=cv2.imread(img1_p2)        
    cv2.imwrite('./pituitaryData/'+modifiedname_p2+'.jpg', img_p2)

for img_filename_n in os.listdir(no_tumor1):
    fname_n=img_filename_n.split(".jpg",1)[0]    
    modifiedname_n="no_"+fname_n    
    img1_n='./Data/Training/no_tumor/'+fname_n+'.jpg'
    img_n=cv2.imread(img1_n)        
    cv2.imwrite('./no_tumor/'+modifiedname_n+'.jpg', img_n)

for img_filename_n2 in os.listdir(no_tumor2):
    fname_n2=img_filename_n2.split(".jpg",1)[0]    
    modifiedname_n2="no_no_"+fname_n2    
    img1_n2='./Data/Testing/no_tumor/'+fname_n2+'.jpg'
    img_n2=cv2.imread(img1_n2)        
    cv2.imwrite('./no_tumor/'+modifiedname_n2+'.jpg', img_n2)

