import cv2

import numpy as np

import mapper

import os

import sys

directory = "D:\CBRN\maps5"

for filename in os.listdir(directory):

    if filename.endswith(".png"): 
        print(filename)

        out_file = filename
        in_file = filename.split('.')[0]+".jpg"


        img2 = cv2.imread(in_file)

        img2 = img2[0:3350, 0:5000]

        #resize image to 1:5

        img = cv2.resize(img2, (0,0), fx=0.2, fy=0.2)

        greyScale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  

        blurImg = cv2.blur(greyScale, (5,5))  

        edgeImg = cv2.Canny(blurImg, 100,200)

        pts = np.argwhere(edgeImg>0)
        y1,x1 = pts.min(axis=0)
        y2,x2 = pts.max(axis=0)

        # crop the region
        cropped = img2[y1*5:y2*5, x1*5:x2*5]
        cv2.imwrite(out_file, cropped)



        # cv2.imshow("edges are",edgeImg)





        # cv2.imwrite(out_file, dst)

        # cv2.waitKey(30)

        cv2.waitKey(1000) 
        cv2.destroyAllWindows() 




        
        

