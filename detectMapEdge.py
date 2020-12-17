import cv2

import numpy as np

import mapper

img2 = cv2.imread("nd-44-10.jpg")

img2 = img2[0:3350, 0:5000]

#resize image to 1:5

img = cv2.resize(img2, (0,0), fx=0.2, fy=0.2)

result = cv2.fastNlMeansDenoisingColored(img,None,20,10,7,21)

edgeImg = cv2.Canny(result, 100,200)

contours,hierarchy=cv2.findContours(edgeImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  
#retrieve the contours as a list, with simple apprximation model

contours=sorted(contours,key=cv2.contourArea,reverse=True)
#the loop extracts the boundary contours of the page

for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*p,True)

    if len(approx)==4:
        target=approx
        break
approx=mapper.mapp(target)
 #find endpoints of the sheet

approx1 = [[(j*5) for j in i] for i in approx]
#scale to get original pts

approx1 = np.float32(approx1)

pts=np.float32([[0,0],[3600,0],[3600,2400],[0,2400]])  
#map to 800*800 target window

op=cv2.getPerspectiveTransform(approx1,pts)  
#get the top or bird eye view effect

dst=cv2.warpPerspective(img2,op,(3600,2400))


# cv2.imshow("Scanned",dst)

cv2.imwrite("nd-44-10.png", dst)


# cv2.imshow("edge detected", edgeImg)

# print(approx)
# print(approx1)

cv2.waitKey(0)
