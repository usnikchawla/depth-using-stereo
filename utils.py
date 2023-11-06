import numpy as np
import cv2
import matplotlib.pyplot as plt

def normalize(uv):

    uv_dash = np.mean(uv, axis=0)
    u_dash ,v_dash = uv_dash[0], uv_dash[1]

    u_cap = uv[:,0] - u_dash
    v_cap = uv[:,1] - v_dash

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    first = np.diag([s,s,1])
    second = np.array([[1,0,-u_dash],[0,1,-v_dash],[0,0,1]])
    T = first.dot(second)

    uv = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(uv.T)).T

    return  x_norm, T


def Estimate(feature_matches):
    
    x1 = feature_matches[:,0:2]
    x2 = feature_matches[:,2:4]
    
    x1_norm, T1 = normalize(x1)
    x2_norm, T2 = normalize(x2)
    
    A=np.zeros((8,9))
    
    for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])
    
    U, S, VT = np.linalg.svd(A, full_matrices=True)
    
    F = VT.T[:, -1]
    F = F.reshape(3,3)
    
    u, s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2,2] = 0
    F = np.dot(u, np.dot(s, vt))
    
    F = np.dot(T2.T, np.dot(F, T1))
            
    return F

def error_F(feature, F): 
    x1,x2 = feature[0:2], feature[2:4]
    x1tmp=np.array([[x1[0], x1[1], 1]]).T
    x2tmp=np.array([[x2[0], x2[1], 1]])

    error = np.dot(x2tmp, np.dot(F, x1tmp))
    
    return np.abs(error)


def fundamentalmatrix(features):
    iter=1000
    error_max=0.01
    
    inlier_thresh=0
    chosen_indices=[]
    chosen_f=0
    
    for i in range(0,iter):
        inlier_indices=[]
        
        rows=features.shape[0]
        random_indices = np.random.choice(rows, size=8)
        features_8=features[random_indices,:]
        
        f_8 = Estimate(features_8)
        for j in range(rows):
            feature = features[j]
            error = error_F(feature, f_8)
            if error < error_max:
                inlier_indices.append(j)
                
        if len(inlier_indices) > inlier_thresh:
            inlier_thresh = len(inlier_indices)
            chosen_indices = inlier_indices
            chosen_f = f_8
            
    filtered_features = features[chosen_indices, :]
    return chosen_f, filtered_features
    
    
def EssentialMatrix(K1, K2, F):
    E = K2.T.dot(F).dot(K1)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E_corrected = np.dot(U,np.dot(np.diag(s),V))
    return E_corrected

def CameraPoses(E):
    """
    Args:
        E (array): Essential Matrix
        K (array): Intrinsic Matrix
    Returns:
        arrays: set of Rotation and Camera Centers
    """
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

   
    R = []
    C = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C

def Points3D(K1, K2, matched_pairs, R2, C2):
    triangulated_points = []
    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    I = np.identity(3)
    P1 = np.dot(K1, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))

    for i in range(len(C2)):
        pts3D = []
        x1 = matched_pairs[:,0:2].T
        x2 = matched_pairs[:,2:4].T

        P2 = np.dot(K2, np.dot(R2[i], np.hstack((I, -C2[i].reshape(3,1)))))

        X = cv2.triangulatePoints(P1, P2, x1, x2)  
        triangulated_points.append(X)
    return triangulated_points

def getPositiveZCount(pts3D, R, C):
    I = np.identity(3)
    P = np.dot(R, np.hstack((I, -C.reshape(3,1))))
    P = np.vstack((P, np.array([0,0,0,1]).reshape(1,4)))
    n_positiveZ = 0
    for i in range(pts3D.shape[1]):
        X = pts3D[:,i]
        X = X.reshape(4,1)
        Xc = np.dot(P, X)
        Xc = Xc / Xc[3]
        z = Xc[2]
        if z > 0:
            n_positiveZ += 1

    return n_positiveZ

def plotMatches(img_1, img_2, matched_pairs, color):

    image_1 = img_1.copy()
    image_2 = img_2.copy()

    
    concat = np.concatenate((image_1, image_2), axis = 1)

    if matched_pairs is not None:
        corners_1_x = matched_pairs[:,0].copy().astype(int)
        corners_1_y = matched_pairs[:,1].copy().astype(int)
        corners_2_x = matched_pairs[:,2].copy().astype(int)
        corners_2_y = matched_pairs[:,3].copy().astype(int)
        corners_2_x += image_1.shape[1]

        for i in range(corners_1_x.shape[0]):
            cv2.line(concat, (corners_1_x[i], corners_1_y[i]), (corners_2_x[i] ,corners_2_y[i]), color, 2)
    
    plt.imshow(concat)
    plt.show()



def EpipolarLines(pts1, pts2, F, image0, image1):
    # set1, set2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]
    lines1, lines2 = [], []
    img_epi1 = image0.copy()
    img_epi2 = image1.copy()

    for i in range(pts1.shape[0]):
        x1 = np.array([pts1[i,0], pts1[i,1], 1]).reshape(3,1)
        x2 = np.array([pts2[i,0], pts2[i,1], 1]).reshape(3,1)

        line2 = np.dot(F, x1)
        

        line1 = np.dot(F.T, x2)
        lines1.append(line1)
    
    
        x1_min = 0
        x1_max = image0.shape[1] -1
        y1_min = -line1[2]/line1[1]
        y1_max = -(line1[2]+line1[0]*x1_max)/line1[1]
        if(y1_min>0 or y1_max>0 ):
            lines1.append(line2)
       
        
        x2_min = 0
        x2_max = image1.shape[1] - 1
        y2_min = -line2[2]/line2[1]
        y2_max = -(line2[2]+line2[0]*x2_max)/line2[1]
        if(y2_min>0 or y2_max>0 ):
            lines2.append(line2)
            
        



        cv2.circle(img_epi2, (int(pts2[i,0]),int(pts2[i,1])), 10, (0,0,255), -1)
        img_epi2 = cv2.line(img_epi2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 0,0), 2)
    

        cv2.circle(img_epi1, (int(pts1[i,0]),int(pts1[i,1])), 10, (0,0,255), -1)
        img_epi1 = cv2.line(img_epi1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 0,0), 2)
        
        
    # concat = np.concatenate((img_epi1, img_epi2), axis = 1)

    # plt.imshow(concat)
    # plt.show()
    return lines1,lines2


def sum_of_abs_diff(pixel_vals_1, pixel_vals_2):
    """
    Args:
        pixel_vals_1 (numpy.ndarray): pixel block from left image
        pixel_vals_2 (numpy.ndarray): pixel block from right image

    Returns:
        float: Sum of absolute difference between individual pixels
    """
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum(abs(pixel_vals_1 - pixel_vals_2))

def compare_blocks(y, x, block_left, right_array, block_size=5):
    """
    Compare left block of pixels with multiple blocks from the right
    image using SEARCH_BLOCK_SIZE to constrain the search in the right
    image.

    Args:
        y (int): row index of the left block
        x (int): column index of the left block
        block_left (numpy.ndarray): containing pixel values within the 
                    block selected from the left image
        right_array (numpy.ndarray]): containing pixel values for the 
                     entrire right image
        block_size (int, optional): Block of pixels width and height. 
                                    Defaults to 5.

    Returns:
        tuple: (y, x) row and column index of the best matching block 
                in the right image
    """
    # Get search range for the right image
    
    SEARCH_BLOCK_SIZE = 56
    
    x_min = max(0, x - SEARCH_BLOCK_SIZE)
    x_max = min(right_array.shape[1]-block_size, x + SEARCH_BLOCK_SIZE)
    #print(f'search bounding box: ({y, x_min}, ({y, x_max}))')
    first = True
    min_sad = None
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y+block_size,
                                  x: x+block_size]
        sad = sum_of_abs_diff(block_left, block_right)
        #print(f'sad: {sad}, {y, x}')
        if first:
            min_sad = sad
            min_index = (y, x)
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                min_index = (y, x)

    return min_index