import cv2 as cv
import numpy as np
import random
from skimage.transform import warp, ProjectiveTransform
import copy

def estimate_fundamental_mat(points1, points2):
    # print(points1)
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
    return Fundamental_mat

def get_start_end_pt(img1, img2, lines, points1, points2):
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, points1, points2):
          
        color = tuple(np.random.randint(0, 255,3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1] ])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1] ])
        # print(pt1[0,0])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1,(x0,y0), 5, color, -1)
        img2 = cv.circle(img2, (x1,y1), 5, color, -1)
    return img1, img2
    # return start_pt, end_pt
    
img1 = cv.imread('artroom/im0.png')
img2 = cv.imread('artroom/im1.png')


sift = cv.SIFT_create()
img1 = cv.resize(img1, (0, 0), fx = 0.5, fy = 0.5)
img2 = cv.resize(img2, (0, 0), fx = 0.5, fy = 0.5)
# cv.imshow('original img1', img1)
# cv.imshow('original img2', img2)
Max_Ransac_iteration = 100
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
best_matches12 = bf.match(descriptors1, descriptors2)
raw_matches12 = sorted(best_matches12, key = lambda x:x.distance)
print(type(raw_matches12))
del raw_matches12[-int(len(raw_matches12)*0.95):-1]
# raw_matches12 = raw_matches12[]
points1 = np.float32([keypoints1[m.queryIdx].pt for m in (raw_matches12)])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in (raw_matches12)])
# points1 = []
# points2 = []
# for m,n in raw_matches12:
#     if m.distance < 0.8*n.distance:
#         points2.append(keypoints2[m.trainIdx].pt)
#         points1.append(keypoints1[m.queryIdx].pt)

array_of_ones = np.matrix(np.ones(len(points1)))
concatenated_matrix1 = np.concatenate((points1, array_of_ones.T), axis=1)
concatenated_matrix2 = np.concatenate((points2, array_of_ones.T), axis=1)
for i in range(Max_Ransac_iteration):
    # estimate the fundamantal mat
    max_inliers= 0
    S = []
    F_best = []
    error = 0
    inliers_count = 0
    absilon = 0.5
    eight_points1 = np.float32([keypoints1[m.queryIdx].pt for m in random.sample(raw_matches12,8)])
    eight_points2 = np.float32([keypoints2[m.trainIdx].pt for m in random.sample(raw_matches12,8)])
    Fundamental_mat = estimate_fundamental_mat(eight_points1, eight_points2)
    for j in range(len(points1)):
        temp = np.dot(concatenated_matrix2[j,:],Fundamental_mat)
        error = np.dot(temp, concatenated_matrix1[j,:].T)
        if error<absilon:
            inliers_count = inliers_count+1
    if max_inliers <  inliers_count:
        max_inliers = inliers_count
        F_best = Fundamental_mat
print("fundamental matrix before", Fundamental_mat)
print("fundamental matrix before", F_best)
F_best, inliers = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC)
print("fundamental matrix afterwords", F_best)
# We select only inlier points
concatenated_matrix1 = concatenated_matrix1[inliers.ravel() == 1]
concatenated_matrix2 = concatenated_matrix2[inliers.ravel() == 1]

# calib_file = open("artroom/calib.txt","r")
# calib_string = calib_file.readline()
# calib_mat = np.matrix(calib_string[5:])
# Essential_mat = np.dot(calib_mat.T,F_best, calib_mat)
# W = np.matrix('0 -1 0; 1 0 0; 0 0 1')
# u_Ess, s_ess, vh_Ess = np.linalg.svd(Essential_mat, full_matrices=True)
# Center_vector = u_Ess[:,-1]
# Rotation_mat = np.dot(u_Ess, W.T, vh_Ess)
# Deter = np.linalg.det(Rotation_mat)

# print("essesntail matrix is \n",Essential_mat)
# print("Fundamental matrix is \n",F_best)
# print("Rotation vec is \n", Rotation_mat)
# print("translation vector is \n", Center_vector)
# print("Determininant of rotation matrix is \n", Deter)


# epilines
lines1 = cv.computeCorrespondEpilines(points2.reshape(-1,1,2),2, F_best)
lines2 = cv.computeCorrespondEpilines(points1.reshape(-1,1,2),1, F_best)
lines1 = lines1.reshape(-1, 3)
lines2 = lines2.reshape(-1, 3)
# print(img1.shape)
img5, img6 = get_start_end_pt((img1), (img2), lines1, points1, points2)
img3, img4 = get_start_end_pt((img2), (img1), lines2, points2, points1)
# cv.imshow('epipolar lines img1', img3)
# cv.imshow('epipolar lines img2', img5)
# good until here
# (Homography_matrix12, status) = cv.findHomography(concatenated_matrix2, concatenated_matrix1, cv.RANSAC)
# (Homography_matrix21, status) = cv.findHomography(concatenated_matrix1, concatenated_matrix2, cv.RANSAC)
_, Homography_matrix12, Homography_matrix21 = cv.stereoRectifyUncalibrated(np.float32(points1), np.float32(points2), F_best, imgSize=(img1.shape[1], img1.shape[0]))
# print(Homography_matrix12.shape)
# print(Homography_matrix21.shape)
# print()
# print(concatenated_matrix1.shape)
newpoints1 = np.dot(concatenated_matrix1, Homography_matrix12)
newpoints2 = np.dot(concatenated_matrix2, Homography_matrix21)
# print("concatenated_matrix1 is ",concatenated_matrix1)
# print("new point is",newpoints1)
# newpoints1 /= newpoints1[2,:]
# newpoints2 /= newpoints2[2,:]
# newpoints1 = newpoints1.T
# newpoints2 = newpoints2.T
# img1_rectified = warp(img1, ProjectiveTransform(matrix=np.linalg.inv(Homography_matrix12)))
# img2_rectified = warp(img1, ProjectiveTransform(matrix=np.linalg.inv(Homography_matrix21)))
img1_rectified = cv.warpPerspective(img1, Homography_matrix12, (img1.shape[1], img1.shape[0]))
img2_rectified = cv.warpPerspective(img2, Homography_matrix21, (img1.shape[1], img1.shape[0]))
# img5_rectified, img6_rectified = get_start_end_pt(img1_rectified, img2_rectified, lines1, newpoints1, newpoints2)
# img3_rectified, img4_rectified = get_start_end_pt(img2_rectified, img1_rectified, lines2, newpoints2, newpoints1)
# print(start_pt1)
# print(end_pt1)
# for s_1,e_1 in start_pt1, end_pt1:(cv.line(img1,s_1,e_1, (255,0,0), thickness=1))

cv.imshow('img1_rectified', img1_rectified)
cv.imshow('img2_rectified', img2_rectified)
cv.waitKey(0)
cv.destroyAllWindows() 