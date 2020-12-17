import cv2

import numpy as np

# Image noise is random variation of brightness or color information in images,
#  and is usually an aspect of electronic noise. 
# It can be produced by the image sensor and circuitry of a scanner or digital camera.
#  ... Image noise is an undesirable by-product of image 
# capture that obscures the desired information.

img2 = cv2.imread("nc-43-03.jpg")


img2 = img2[0:3350, 0:5000]

#resize image to 1:5


img = cv2.resize(img2, (0,0), fx=0.2, fy=0.2)

# img1 = cv2.blur(img, (3,3))
# contrastImg = cv2.addWeighted(img1,1.5, np.zeros(img.shape, img.dtype), 0, 0)

# contrastImg2 = cv2.addWeighted(img,1.5, np.zeros(img.shape, img.dtype), 0, 0)
# img2 = cv2.blur(contrastImg2, (3,3))



result = cv2.fastNlMeansDenoisingColored(img,None,20,10,7,21)

result1 = cv2.fastNlMeansDenoisingColored(img,None,35,10,7,21)

edgeImg = cv2.Canny(result, 100,200)

edgeImg1 = cv2.Canny(result1, 100, 200)

## find the non-zero min-max coords of canny
pts = np.argwhere(edgeImg1>0)
y1,x1 = pts.min(axis=0)
y2,x2 = pts.max(axis=0)

## crop the region
# cropped = img2[y1*5:y2*5, x1*5:x2*5]
# cv2.imwrite("cropped3.png", cropped)

# cv2.imshow("original img", img)
# cv2.imshow("original contrast img", contrast_img)
# cv2.imshow("noise reduced img", result)
# cv2.imshow("noise reduced contrast img", result1)
cv2.imshow("low str noise", edgeImg)
cv2.imshow("high str noise", edgeImg1)

cv2.waitKey(0)
