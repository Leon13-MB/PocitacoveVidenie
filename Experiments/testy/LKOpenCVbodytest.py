import numpy as np
import cv2 as cv
import argparse
import pandas as pd
#https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4
filename = str('./Python/videoforlk.mp4')
df_points=pd.read_excel('./Python/BodyOpenCV.xlsx', index=False)

cap = cv.VideoCapture(filename)
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
while(1):
    frameid=cap.get(1)
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    if(int(frameid)==10):
        for p in good_new:
            df2= pd.DataFrame({
                    'points10': pd.Series(p)
                })
            df_points=df_points.append(df2,ignore_index=True,sort=False)
            with pd.ExcelWriter('./Python/BodyOpenCV.xlsx') as writer:
                    df_points.to_excel(writer, sheet_name='Points', index=False)
    if(int(frameid)==100):
        for p in good_new:
            df2= pd.DataFrame({
                    'points100': pd.Series(p)
                })
            df_points=df_points.append(df2,ignore_index=True,sort=False)
            with pd.ExcelWriter('./Python/BodyOpenCV.xlsx') as writer:
                    df_points.to_excel(writer, sheet_name='Points', index=False)
    if(int(frameid)==500):
        for p in good_new:
            df2= pd.DataFrame({
                    'points500': pd.Series(p)
                })
            df_points=df_points.append(df2,ignore_index=True,sort=False)
            with pd.ExcelWriter('./Python/BodyOpenCV.xlsx') as writer:
                    df_points.to_excel(writer, sheet_name='Points', index=False)
    if(int(frameid)==900):
        for p in good_new:
            df2= pd.DataFrame({
                    'points900': pd.Series(p)
                })
            df_points=df_points.append(df2,ignore_index=True,sort=False)
            with pd.ExcelWriter('./Python/BodyOpenCV.xlsx') as writer:
                    df_points.to_excel(writer, sheet_name='Points', index=True)
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