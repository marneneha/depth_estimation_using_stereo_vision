import cv2 as cv
import numpy as np


img1 = cv.imread('artroom/im0.png')
img2 = cv.imread('artroom/im1.png')
sift = cv.SIFT_create()
img1 = cv.resize(img1, (0, 0), fx = 0.5, fy = 0.5)
img2 = cv.resize(img2, (0, 0), fx = 0.5, fy = 0.5)
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
best_matches12 = bf.match(descriptors1, descriptors2)
raw_matches12 = sorted(best_matches12, key = lambda x:x.distance)
points1 = np.float32([keypoints1[m.queryIdx].pt for m in raw_matches12])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in raw_matches12])
array_of_ones = np.matrix(np.ones(len(points1)))
concatenated_matrix1 = np.concatenate((points1, array_of_ones.T), axis=1)
concatenated_matrix2 = np.concatenate((points2, array_of_ones.T), axis=1)
A_mat = np.matrix(np.ones(9))
for i in range(len(points1)):
    A_mat_row = np.concatenate((concatenated_matrix1[i,0]*concatenated_matrix2[i,:], concatenated_matrix1[i,1]*concatenated_matrix2[i,:], concatenated_matrix2[i,:]),axis=1)
    A_mat = np.concatenate((A_mat, A_mat_row),axis=0)
A_mat = np.delete(A_mat, 0,0)
u, s, vh = np.linalg.svd(A_mat, full_matrices=True)
element_vec = vh[:,-1]
Fundamental_mat = element_vec.reshape((3,3))

calib_file = open("artroom/calib.txt","r")
# print(calib_file.readline())
calib_string = calib_file.readline()
calib_mat = np.matrix(calib_string[5:])
# print(calib_mat)
Essential_mat = np.dot(calib_mat.T,Fundamental_mat, calib_mat)
print(Essential_mat)

W = np.matrix('0 -1 0; 1 0 0; 0 0 1')
u_Ess, s_ess, vh_Ess = np.linalg.svd(Essential_mat, full_matrices=True)
Center_vector = u_Ess[:,-1]
Rotation_mat = np.dot(u_Ess, W.T, vh_Ess)
print(Center_vector)
print(np.linalg.det(Rotation_mat))

cv.imshow('image0', img1)
cv.imshow('image1', img2)
cv.waitKey(0)
cv.destroyAllWindows() 