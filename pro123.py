import cv2
from cv2 import IMWRITE_PNG_BILEVEL
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score as asc
from PIL import Image
import PIL.ImageOps
import time

x=np.load('image.npz')['arr_0']
y=pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())

classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)

xtrain,xtest,ytrain,ytest=tts(x,y,train_size=7500,test_size=2500,random_state=9)
xtrain=xtrain/255.0
xtest=xtest/225.0

clf=lr(solver='saga',multi_class='multinomial').fit(xtrain,ytrain)

pred=clf.predict(xtest)
print(asc(ytest,pred))

cam=cv2.VideoCapture(0)

while(True):
    try:
        ret,frame=cam.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upperleft=(int(width/2-50),int(height/2-50))
        bottomright=(int(width/2+50),int(height/2+50))
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi=gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        im_pil=Image.fromarray(roi)
        im_bw=im_pil.convert('L')
        im_bw_resize=im_bw.resize((28,28),Image.ANTIALIAS)
        im_bw_resize_inverted=PIL.ImageOps.invert(im_bw_resize)
        min_pixel=np.percentile(im_bw_resize_inverted,20)
        im_bw_resize_inverted_scale=np.clip(im_bw_resize_inverted-min_pixel,0,255)
        max_pixel=np.max(im_bw_resize_inverted)
        im_bw_resize_inverted_scale=np.asarray(im_bw_resize_inverted_scale)/max_pixel
        testsample=np.array(im_bw_resize_inverted_scale).reshape(1,784)
        testpred=clf.predict(testsample)
        print("Predicted class is: ",testpred)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    except Exception as e:
        pass
    
cam.release()
cv2.destroyAllWindows()