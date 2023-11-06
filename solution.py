"""
In this script, we are going to implement the concept of Stereo Vision. We will be
given 3 different datasets, each of them contains 2 images of the same scenario but
taken from two different camera angles. By comparing the information about a
scene from 2 vantage points, we can obtain the 3D information by examining the
relative positions of objects.

"""


import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *
from tqdm import *
import sys

data=sys.argv[1]
img1=cv2.imread(f"data/{data}/im0.png")
img2=cv2.imread(f"data/{data}/im1.png")

img1_gray=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
k=[]
baseline=0
if data=="curule":
    K=np.array([[1758.23,0,977.42],[0,1758.23,552.15],[0,0,1]])
    baseline = 88.39
    
elif data=="octagon":
    K=np.array([[1742.11,0,804.90],[0,1742.11,541.22],[0,0,1]])
    baseline = 221.76
    
elif data=="pendulum":
    
    K=np.array([[1729.05,0,-364.24],[0,1729.05,552.22],[0,0,1]])
    baseline = 221.76

else:
    print("Enter a valid value and rerun the script")

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1_gray,None)
kp2, des2 = sift.detectAndCompute(img2_gray,None)


bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
best_matches = bf.match(des1,des2)

matches = sorted(best_matches, key = lambda x:x.distance)

choosen_matches=matches[0:200]



kp1 = np.float32([kp.pt for kp in kp1])
kp2 = np.float32([kp.pt for kp in kp2])

#Creating a list of coordinates for features that match in both the images.
pts1 = np.array([kp1[m.queryIdx].tolist() for m in choosen_matches])
pts2 = np.array([kp2[m.trainIdx].tolist() for m in choosen_matches])
features=np.concatenate((pts1,pts2),axis=1)



F, matched_inliers=fundamentalmatrix(features)
print("Fundamental Matrix: ", F)
#plotMatches(img1,img2,matched_inliers,(255,0,0))

matched_inliers=np.concatenate((pts1,pts2),axis=1)

E = EssentialMatrix(K, K, F)

print("Essential Matrix: ",E)

R2, C2 = CameraPoses(E)

triangulated_points = Points3D(K, K, matched_inliers, R2, C2)

z_1 = []
z_2 = []

R1 = np.identity(3)
C1 = np.zeros((3,1))
for i in range(len(triangulated_points)):
    pt3D = triangulated_points[i]  
    pt3D = pt3D/pt3D[3, :]
    x = pt3D[0,:]
    y = pt3D[1, :]
    z = pt3D[2, :]   

    z_2.append(getPositiveZCount(pt3D, R2[i], C2[i]))
    z_1.append(getPositiveZCount(pt3D, R1, C1))
    
z_1 = np.array(z_1)
z_2 = np.array(z_2)
thresh = int(triangulated_points[0].shape[1] / 2)
idx = np.intersect1d(np.where(z_1 > thresh), np.where(z_2 > thresh))

R = R2[idx[0]]
C = C2[idx[0]]

print("Rotation matrx: ", R)
print("translation matrix(upto a scale): ",C)

pts1=matched_inliers[:,0:2]
pts2=matched_inliers[:,2:]

h1, w1,_= img1.shape
h2, w2,_ = img2.shape

_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))

img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))


pts1 = cv2.perspectiveTransform(np.float32(pts1.reshape(-1, 1, 2)), H1).reshape(-1,2)
pts2 = cv2.perspectiveTransform(np.float32(pts2.reshape(-1, 1, 2)), H2).reshape(-1,2)

H2_inv_trans=np.linalg.inv(H2).T
H1_inv=np.linalg.inv(H1)

F_new=np.dot(H2_inv_trans,np.dot(F,H1_inv))

lines1,lines2=EpipolarLines(pts1,pts2,F_new,img1_rectified,img2_rectified)


img1_rectified=cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
img2_rectified=cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

img1_rectified = cv2.resize(img1_rectified, (int(img1_rectified.shape[1] / 4), int(img1_rectified.shape[0] / 4)))
img2_rectified = cv2.resize(img2_rectified, (int(img2_rectified.shape[1] / 4), int(img2_rectified.shape[0] / 4)))

concat = np.concatenate((img1_rectified, img2_rectified), axis = 1)


h,w=img1_rectified.shape
disparity_map = np.zeros((h, w))
BLOCK_SIZE=7
for y in tqdm(range(0, h-BLOCK_SIZE)):
    for x in range(0, w-BLOCK_SIZE):
        block_left = img1_rectified[y:y + BLOCK_SIZE,
                                x:x + BLOCK_SIZE]
        min_index = compare_blocks(y, x, block_left,
                                    img2_rectified,
                                    block_size=BLOCK_SIZE)
        
        disparity_map[y, x] = abs(min_index[1] - x)



f=K[0,0] 
depth = (baseline * f) / (disparity_map + 1e-10)
depth[depth > 50000] = 50000
disparity_map = np.uint8(disparity_map * 255 / np.max(disparity_map))
depth = np.uint8(depth * 255 / np.max(depth)) 




plt.subplot(1,2,1)
plt.imshow(depth,cmap='hot')
plt.subplot(1,2,2)
plt.imshow(disparity_map,cmap='gray',interpolation='nearest')
plt.show()
