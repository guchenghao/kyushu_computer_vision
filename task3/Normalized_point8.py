import numpy as np
import cv2
from numpy import linalg

dir1 = "./images/cups1.JPG"
dir2 = "./images/cups2.JPG"

def get_points(img_dir1,img_dir2):
    points1 = []
    points2 = []

    def OnMouseAction1(event,x,y,flags,param):

        if event == cv2.EVENT_LBUTTONDOWN:
            print("from camera1:(%s,%s)"%(x,y))
            points1.append([x,y])
        elif event==cv2.EVENT_RBUTTONDOWN :
            print("camera1 finish")
            cv2.destroyWindow("image1")
            
    def OnMouseAction2(event,x,y,flags,param):

        if event == cv2.EVENT_LBUTTONDOWN:
            print("from camera2:(%s,%s)"%(x,y))
            points2.append([x,y])
        elif event==cv2.EVENT_RBUTTONDOWN :
            print("camera2 finish")
            cv2.destroyWindow("image2")

    img1 = cv2.imread(img_dir1)
    img2 = cv2.imread(img_dir2)
    cv2.namedWindow("image1",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image1", 540, 480)
    cv2.setMouseCallback('image1',OnMouseAction1)     
    cv2.namedWindow("image2",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image2", 540, 480)
    cv2.setMouseCallback('image2',OnMouseAction2)
    cv2.imshow('image1',img1)
    cv2.imshow('image2',img2)
    cv2.waitKey(80000)
    
         
    
    return (np.array(points1),np.array(points2))


def normalization(pts):

    centroid = np.array([np.mean(pts[:, 0]), np.mean(pts[:, 1])])

    distance = 0
    for index in range(pts[:, 0].size):
        distance += np.linalg.norm(centroid - pts[index, :])
    meanDistance = distance / pts[:, 0].size


    T = np.array([[np.sqrt(2) / meanDistance, 0, -centroid[0] * np.sqrt(2) / meanDistance],
                  [0, np.sqrt(2) / meanDistance, -centroid[1]
                   * np.sqrt(2) / meanDistance],
                  [0, 0, 1]])

    # Normalize points
    result = np.ones((pts[:, 0].size, 2), dtype=float)
    for index in range(pts[:, 0].size):
        current = np.dot(T, np.array([pts[index, 0], pts[index, 1], 1]))
        result[index, 0] = current[0]
        result[index, 1] = current[1]

    return result, T


def FM_by_normalized_8_point(pts1, pts2):

    # Normalizing points
    normalizedPts1, T1 = normalization(pts1)
    normalizedPts2, T2 = normalization(pts2)

    print(normalizedPts1)
    print(normalizedPts1.shape)

    u1 = normalizedPts1[:, 0]
    v1 = normalizedPts1[:, 1]
    u2 = normalizedPts2[:, 0]
    v2 = normalizedPts2[:, 1]
    ones = np.ones(normalizedPts1[:, 0].size, int)

    # Create the constraint matrix
    A = np.array([u1 * u2, u1 * v2, u1, v1 * u2, v1 *
                 v2, v1, u2, v2, ones]).transpose()
    U, D, V = np.linalg.svd(A)


    FMatrix = np.reshape(V.transpose()[:, 8], (3, 3)).transpose()
    print(FMatrix)
    # Enforce rank2 constraint
    U, D, V = np.linalg.svd(FMatrix)
    FMatrix = U @ np.diag(np.array([D[0], D[1], 0])) @ V
    print(FMatrix)
    # De-normalize
    FMatrix = T2.transpose() @ FMatrix @ T1
    print(FMatrix)

    # normalize the fundamental matrix so that the value on [2, 2] is 1
    FMatrix = np.true_divide(FMatrix, FMatrix[2, 2])
    print(FMatrix.shape)
    print(pts1.shape)

    _, eig_list_e1 = linalg.eig(np.dot(FMatrix.T, FMatrix))
    e1 = eig_list_e1.T[-1]/eig_list_e1.T[-1][2]
    _, eig_list_e2 = linalg.eig(np.dot(FMatrix, FMatrix.T))
    e2 = eig_list_e2.T[-1]/eig_list_e2.T[-1][2]

    return (FMatrix, e1, e2)


def draw_lines(point1,point2,e1,e2,dir1,dir2):
    one = np.ones((len(point1),1),"float")
    point1_ = np.concatenate((point1,one),axis=1)
    point2_ = np.concatenate((point2,one),axis=1)
    line1 = []
    line2 = []
    
    for p1 , p2 in zip(point1_,point2_):
        line1.append(np.cross(e1,p1))
        line2.append(np.cross(e2,p2))
        
    i = 0
    cv2.namedWindow('epilines1',cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow("image1", 540, 480)
    img1 = cv2.imread(dir1)
    for l1,p1 in zip(line1,point1):
        i += 1
        aa = tuple(p1.tolist())
        print("circle")
        cv2.circle(img1, aa, 15, (255,0,16), -1)
        cv2.putText(img1,'%s'%i,(p1[0],p1[1]+10),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),9)
        print("lines")
        cv2.line(img1, (15, int(-l1[2] / l1[1])),(img1.shape[1], int(-(l1[2] + l1[0] * img1.shape[1]) / l1[1])),(0,255,0),5)
    cv2.imshow('epilines1',img1)
    cv2.imwrite("./results/images/2-1_lines.jpg", img1)
    cv2.waitKey(0)
    
    j = 0
    cv2.namedWindow('epilines2',cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow("image1", 540, 480)
    img2 = cv2.imread(dir2)
    for l2,p2 in zip(line2,point2):
        j += 1
        bb = tuple(p2.tolist())
        cv2.circle(img2, bb, 15, (255,0,16), -1)
        cv2.putText(img2,'%s'%j,(p2[0],p2[1]+10),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),9)
        cv2.line(img2, (0, int(-l2[2] / l2[1])),(img2.shape[1], int(-(l2[2] + l2[0] * img2.shape[1]) / l2[1])), (0,255,0),5)
    cv2.imshow('epilines2',img2)
    cv2.imwrite("./results/2-2_lines.jpg", img2)
    cv2.waitKey(0)


a, b = get_points(dir1, dir2)
print(a)
print(b)

F, e1, e2 = FM_by_normalized_8_point(a, b)

print(F)
print(e1, e2)

draw_lines(a,b,e1,e2,dir1,dir2)