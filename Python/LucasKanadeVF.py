import numpy as np
import cv2 as cv
from scipy import signal

#https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4
filename = str('./Python/videoforlk.mp4')

#parametre pre najdenie bodov na okrajoch objektov
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

vc=0
if(vc):
    capture= cv.VideoCapture(0)
else:
    capture= cv.VideoCapture(filename)
def LucasKanadeCalc(prev_frame, new_frame, points):
    #Masky pre konvoluciu
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])
    #valid-podla velkosti dimenzii
    mode = 'valid'
    #ziskanie derivacii fx,fy a ft
    fx = signal.convolve2d(prev_frame, kernel_x, mode=mode)
    fy = signal.convolve2d(prev_frame, kernel_y, mode=mode)
    ft = signal.convolve2d(new_frame, kernel_t, mode=mode) + signal.convolve2d(prev_frame, -kernel_t, mode=mode)
    #velkost okolia
    w=15
    new_points = np.zeros(points.shape)
    k = 0
    for p in points:
        i = int(p.item(1))
        j = int(p.item(0))
        #vyberame funkcie z okolia bodu
        Ix= fx[i-w:i+w, j-w:j+w].flatten()
        Iy= fy[i-w:i+w, j-w:j+w].flatten()
        It= ft[i-w:i+w, j-w:j+w].flatten()
        #ziskanie b,A
        b = It
        A = np.vstack((Ix, Iy)).T
        #vektor obsahujuci u,v
        nu = np.matmul(np.linalg.pinv(A), b)
        #nu[0] = u
        #nu[1] = v
        
        item1 = float(p.item(0)+(nu[0]))
        item2 = float(p.item(1)+(nu[1]))
        arraytemp = np.array([[item1,item2]])
        new_points[k] = arraytemp
        k=k+1
    k=0
        
    return np.array(new_points)

m, frame1 = capture.read()
height,width,_ = frame1.shape
im1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
#Najdenie bodov v prvom frame
points = cv.goodFeaturesToTrack(im1, mask = None, **feature_params)
#maska pre vykreslovanie ciar
mask = np.zeros_like(frame1)
while(capture.isOpened()):
    frameid = capture.get(1)
    print(frameid)
    check, frame2 = capture.read()
    im2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    #prepocet LK algoritmu
    newp=LucasKanadeCalc(im1,im2,points)
    
    #vykreslenie bodov a ciar
    for _,(n,p) in enumerate(zip(newp, points)):
        a = int(n.item(0))
        b = int(n.item(1))
        c = int(p.item(0))
        d = int(p.item(1))
        mask = cv.line(mask, (a,b),(c,d), (0,0,255), 2)
        frame2=cv.circle(frame2,(c,d),5,(0,0,255),2)

    img = cv.add(frame2,mask)
    cv.imshow('frame',img)
    
    # if(int(frameid)==120):
    #     cv.imwrite("./Python/120framew9.jpg",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    if(int(frameid) == 910):
        break
    im1=im2.copy()
    points=np.asarray(newp).reshape(-1,1,2)
capture.release()
cv.destroyAllWindows()
