import numpy as np
import cv2 as cv
import argparse
import time
import pandas as pd
#https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4
filename = str('./Python/videoforlk.mp4')
cap = cv.VideoCapture(filename)
df_cas=pd.read_excel('./Python/OpenCVtest2.xlsx', index=False)
counter=0
print('pred',df_cas)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
temp = []
def cyklus(old_gray,p0,mask,o,df_cas,cap,temp,fn):
    while(1):
        frameid=cap.get(1)
        if(o==20):
            df2= pd.DataFrame({
                'opencvf': pd.Series(temp)
            })
            df_cas=df_cas.append(df2,ignore_index=True)
            print('po:',df_cas)
            with pd.ExcelWriter('./Python/OpenCVtest2.xlsx') as writer:
                df_cas.to_excel(writer, sheet_name='OpenCV', index=False)
            break
        ret,frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        start=time.time()
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        elap_time = time.time()*1000 - start*1000
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv.add(frame,mask)
        cv.imshow('frame',img)
        if(int(frameid)==912):
            break
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        if(int(frameid)==10):
            temp.append(elap_time)
            print(o)
            cap.release()
            cap = cv.VideoCapture(fn)
            ret, old_frame = cap.read()
            old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
            p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
            mask = np.zeros_like(old_frame)
            o=o+1
            cyklus(old_gray,p0,mask,o,df_cas,cap,temp,fn)
if(counter==0):
    counter=counter+1
    cyklus(old_gray,p0,mask,counter-1,df_cas,cap,temp,filename)
cap.release()