
import numpy as np
import cv2
import glob
import copy


#Adaptive non-maximal suppression
def ANMS(grayscale, N_best):
    corners = cv2.goodFeaturesToTrack(grayscale, 100000, 0.001, 10, useHarrisDetector=False)
    gray = grayscale

    N_strong = corners.shape[0]
    r = np.inf * np.ones([N_strong,3])
    ED = 0

    for i in range(N_strong):    
        for j in range(N_strong):
            xi = int(corners[i,:,0])
            yi = int(corners[i,:,1])
            xj = int(corners[j,:,0])
            yj = int(corners[j,:,1])
            if gray[yi,xi]>gray[yj,xj]:
                ED = (xj-xi)**2 +(yj-yi)**2
            if ED < r[i,0]:
                r[i, 0] = ED
                r[i, 1] = xi
                r[i, 2] = yi

    r_i = r[np.argsort(-r[:, 0])] 
    best_c = r_i[:N_best,:] 
    return best_c

def feature_descriptor(grayscale, best_c):
    patch_size=40
    a,b = best_c.shape
    grayscale = np.pad(grayscale, int(patch_size), 'constant', constant_values=0)
    ft_desc = np.array(np.zeros((int((patch_size/5)**2),1)))

    for i in range(a):
        patch = grayscale[int(best_c[i][2]+(patch_size/2)):int(best_c[i][2]+(3*patch_size/2)),int(best_c[i][1]+(patch_size/2)):int(best_c[i][1]+(3*patch_size/2))]

        gaussian_blur= cv2.GaussianBlur(patch, (5,5), 0)
        sub_sample = cv2.resize(gaussian_blur, (8,8))
        
        sub_sample= sub_sample.reshape(int((patch_size/5)**2),1)
        sub_sample=(sub_sample-np.mean(sub_sample))/np.std(sub_sample)  
 
        ft_desc = np.dstack((ft_desc,sub_sample))

    return ft_desc[:,:,1:]


def MatchFeatures(vec1, vec2, corner1, corner2):
    p,x,q = vec1.shape
    m,y,n = vec2.shape
    matchPairs = []
    q = int(min(q,n))
    n = int(max(q,n))
    for i in range(q):
        match = {}
        for j in range(n):
            ssd=np.linalg.norm((vec1[:,:,i]-vec2[:,:,j]))**2

            match[ssd] = [corner1[i,:],corner2[j,:]]

        S = sorted(match)
        if S[0]/S[1] < 0.7:
            pairs = match[S[0]]
            matchPairs.append(pairs)

    return matchPairs  


def plot_matches(img1, img2, matchPairs, num):
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))

    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    image_1 = new_img.copy()

    for i in range(len(matchPairs)):
        cv2.line(image_1, (int(matchPairs[i][0][1]), int(matchPairs[i][0][2])), (int(matchPairs[i][1][1]+img1.shape[1]), int(matchPairs[i][1][2])), (0,0,255), 1)
        cv2.circle(image_1,(int(matchPairs[i][0][1]), int(matchPairs[i][0][2])),3,(255,255,0),1)
        cv2.circle(image_1,(int(matchPairs[i][1][1])+img1.shape[1], int(matchPairs[i][1][2])),3,(240,255,0),1)
    
    cv2.imwrite('/home/ardangle/Downloads/P1_CV/YourDirectoryID_p1/Phase1/Results/random/plotted'+str(num)+'.jpg', image_1)
    cv2.imshow("plotted", image_1)
    cv2.waitKey(0)

def plot_matches2(img1, img2, matchPairs, num):
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))


    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    image_1 = new_img.copy()

    for i in range(len(matchPairs)):
        cv2.line(image_1, (int(matchPairs[i][0][1]), int(matchPairs[i][0][2])), (int(matchPairs[i][1][1]+img1.shape[1]), int(matchPairs[i][1][2])), (0,0,255), 1)
        cv2.circle(image_1,(int(matchPairs[i][0][1]), int(matchPairs[i][0][2])),3,(255,255,0),1)
        cv2.circle(image_1,(int(matchPairs[i][1][1])+img1.shape[1], int(matchPairs[i][1][2])),3,(240,255,0),1)
    
    cv2.imwrite('/home/ardangle/Downloads/P1_CV/YourDirectoryID_p1/Phase1/Results/random/ransac_plotted'+str(num)+'.jpg', image_1)
    cv2.imshow("plotted", image_1)
    cv2.waitKey(0)

    
def RANSAC(pairs,t):
    thresh=30.0 
    print("pairs........", len(pairs))
    H_new = np.zeros((3,3))
    max = 0

    for j in range(3000):
        index = []
        pts = [np.random.randint(0,len(pairs)) for i in range(4)]
        p1 = np.array([[pairs[pts[0]][0][1:3]],[pairs[pts[1]][0][1:3]],[pairs[pts[2]][0][1:3]],[pairs[pts[3]][0][1:3]]],np.float32)
        p2 = np.array([[pairs[pts[0]][1][1:3]],[pairs[pts[1]][1][1:3]],[pairs[pts[2]][1][1:3]],[pairs[pts[3]][1][1:3]]],np.float32)

        H = cv2.getPerspectiveTransform(p1, p2)
        inliers = 0
        
        for ind in range(len(pairs)):
            a = np.array(pairs[ind][1][1:3])
            b = np.array(pairs[ind][0][1:3])
            p = np.matmul(H, np.array([b[0],b[1],1]))
            if p[2] == 0:
                p[2] = 0.00001
            p_x = p[0]/p[2]
            p_y = p[1]/p[2]
            p = np.array([p_x,p_y])
            p = np.float32([point for point in p])
            if (np.linalg.norm(a-p)) < thresh:
                inliers += 1
                index.append(ind)

        u = []
        v = []
        if max < inliers:
            max = inliers
            [u.append([pairs[i][0][1:3]]) for i in index]
            [v.append([pairs[i][1][1:3]]) for i in index]
            H_new,df = cv2.findHomography(np.float32(u),np.float32(v))

            if inliers > t*len(pairs):
                break

    match_pair = [pairs[i] for i in index]
    print("ransac pairs......", len(match_pair))

    return H_new, match_pair


def stitch(image, homography, image2_shape):
    h, w, k = np.shape(image)
    randomH = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    calcH = np.dot(homography, randomH)

    row_y = calcH[1] / calcH[2]
    row_x = calcH[0] / calcH[2]

    new_mat = np.array([[1, 0, -1 * min(row_x)], [0, 1, -1 * min(row_y)], [0, 0, 1]])
    homography = np.dot(new_mat, homography)

    h = int(round(max(row_y) - min(row_y)))+image2_shape[0]
    w = int(round(max(row_x) - min(row_x)))+ image2_shape[1]
    size = (h,w)

    warp = cv2.warpPerspective(src=image, M=homography, dsize=size)
    return warp, int(min(row_x)), int(min(row_y))


def combine(img1, img2, num, t, ANMS_corners):

    gimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
    gimg1 = np.float32(gimg1)

    gimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
    gimg2 = np.float32(gimg2)

    i1 = copy.deepcopy(img1)
    i2 = copy.deepcopy(img2)
    i3 = copy.deepcopy(img1)   
    i4 = copy.deepcopy(img2) 
    i5 = copy.deepcopy(img1) 
    i6 = copy.deepcopy(img2)

    # ANMS_corners=500
    corners1 = ANMS(gimg1,ANMS_corners)
    for corner in corners1:
        z,x3,y3 = corner.ravel()  
        x3 = int(x3)
        y3 = int(y3)
        cv2.circle(i5,(x3,y3),3,(0,100,255),-1)
    cv2.imwrite('/home/ardangle/Downloads/P1_CV/YourDirectoryID_p1/Phase1/Results/random/corners1'+str(num)+'.png', i5)


    corners2 = ANMS(gimg2,ANMS_corners)
    for corner in corners2:
        z,x3,y3 = corner.ravel()  
        x3 = int(x3)
        y3 = int(y3)
        cv2.circle(i6,(x3,y3),3,(0,100,255),-1)
    cv2.imwrite('/home/ardangle/Downloads/P1_CV/YourDirectoryID_p1/Phase1/Results/random/corners2'+str(num)+'.png', i6)


    features1 = feature_descriptor(gimg1,corners1)
    features2 = feature_descriptor(gimg2,corners2)

    match = MatchFeatures(features1,features2,corners1, corners2)
    plot_matches(i1, i2, match, num)
    H , pairs = RANSAC(match, t)
    # print(H)
    plot_matches2(i3, i4, pairs, num)

    return H, pairs



def main():
    images=[]
    BasePath="/home/ardangle/Downloads/P1_CV/YourDirectoryID_p1/Phase1/Data/Train/test2/"
   
    images = [cv2.imread(file) for file in sorted(glob.glob(str(BasePath)+'/*.jpg'))]
    num=0
    t=0.3
    im2=images[0]
    N_best=500
    im2= cv2.resize(im2, (300,500), interpolation = cv2.INTER_AREA)
    for i, im in enumerate(images[1:]):
        im= cv2.resize(im, (300,500), interpolation = cv2.INTER_AREA)
        H, pairs= combine(im2, im, num, t, N_best) 
        num=num+1       

        if i>4:
            imgholder, offsetX, offsetY = stitch(im,np.inv(H),im2.shape)
            print("imgholder", imgholder.shape)
            print("y===", im.shape[0]+abs(offsetY))
            print("x===", im.shape[1]+abs(offsetX))
            for y in range(abs(offsetY),im.shape[0]+abs(offsetY)):
                for x in range(abs(offsetX),im.shape[1]+abs(offsetX)):
                    img2_y = y - abs(offsetY) 
                    img2_x = x - abs(offsetX)
                    imgholder[y+150,x+150,:] = im[img2_y,img2_x,:]
                    N_best=N_best-100
                    t=t-0.05

        else:
            imgholder, offsetX, offsetY = stitch(im2,H,im.shape)
            print("imgholder", imgholder.shape)
            print("y===", im.shape[0]+abs(offsetY))
            print("x===", im.shape[1]+abs(offsetX))   
            for y in range(abs(offsetY),im.shape[0]+abs(offsetY)):
                for x in range(abs(offsetX),im.shape[1]+abs(offsetX)):
                    img2_y = y - abs(offsetY) 
                    img2_x = x - abs(offsetX)
                    imgholder[y,x,:] = im[img2_y,img2_x,:]

        im2 = imgholder

    cv2.imshow('blur_pano.png',im2)
    cv2.imwrite("/home/ardangle/Downloads/P1_CV/YourDirectoryID_p1/Phase1/Results/random/result.jpg", im2)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
