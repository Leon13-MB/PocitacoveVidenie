import numpy as np
import cv2 as cv
from scipy import signal
import time
import pandas as pd
#https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4
filename = str('./Python/videoforlk.mp4')
df_cas=pd.read_excel('./Python/VlastnaFunkciaTest.xlsx', index=False)
counter=0
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
    w=30
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


temp=[]
m, frame1 = capture.read()
height,width,_ = frame1.shape
im1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
#Najdenie bodov v prvom frame
points = cv.goodFeaturesToTrack(im1, mask = None, **feature_params)
#maska pre vykreslovanie ciar
mask = np.zeros_like(frame1)
def cyklus(im1,points,mask,o,df_cas,capture,temp,fn):
    while(capture.isOpened()):
        frameid=capture.get(1)
        if(o==20):
            df2= pd.DataFrame({
                'vlastnaf': pd.Series(temp)
            })
            df_cas=df_cas.append(df2,ignore_index=True)
            print('po:',df_cas)
            with pd.ExcelWriter('./Python/VlastnaFunkciaTest.xlsx') as writer:
                df_cas.to_excel(writer, sheet_name='OpenCV', index=False)
            break
        check, frame2 = capture.read()
        #vykreslenie bodov z prveho frame-u
        im2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        
        #prepocitanie bodov v dalsich frame-och pomocou lucas-kanade algoritmu
        start = time.time()
        newp=LucasKanadeCalc(im1,im2,points)
        end = time.time()
        elapsed_time= end*1000 - start*1000

        for i,(n,p) in enumerate(zip(newp, points)):
            a = int(n.item(0))
            b = int(n.item(1))
            c = int(p.item(0))
            d = int(p.item(1))
            mask = cv.line(mask, (a,b),(c,d), (0,0,255), 2)
            frame2=cv.circle(frame2,(c,d),5,(0,0,255),2)
        #frame+ciary+body
        img = cv.add(frame2,mask)
        cv.imshow('frame',img)
        if(int(frameid)==912):
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        im1=im2.copy()
        points=np.asarray(newp).reshape(-1,1,2)
        if(int(frameid)==10):
            temp.append(elapsed_time)
            print(o)
            capture.release()
            capture = cv.VideoCapture(fn)
            m, frame1 = capture.read()
            im1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            points = cv.goodFeaturesToTrack(im1, mask = None, **feature_params)
            mask = np.zeros_like(frame1)
            o=o+1
            cyklus(im1,points,mask,o,df_cas,capture,temp,fn)


if(counter==0):
    counter=counter+1
    cyklus(im1,points,mask,counter-1,df_cas,capture,temp,filename)
capture.release()
cv.destroyAllWindows()
