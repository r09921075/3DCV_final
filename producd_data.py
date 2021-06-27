import cv2
import numpy as np
video = cv2.VideoCapture("data/video_resource/car.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
# frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
# size = np.array([int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))])

# videoWriter = cv2.VideoWriter('trans.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, size)  
success, frame = video.read()  
# frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
index = 0
ratio = 0.1
# path_in = 'data/resource_img/f1/'
# path_out = 'data/need_img/f1/'
path = 'data/gan/'
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.createBackgroundSubtractorMOG2()
while success :  
    # fgmask = fgbg.apply(frame)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('frame',fgmask)
    # cv2.putText(frame, 'fps: ' + str(fps), (0,int(1*ratio*size[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 5)
    # cv2.putText(frame, 'count: ' + str(frameCount), (0, int(2*ratio*size[1])), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,255), 5)
    # cv2.putText(frame, 'frame: ' + str(index), (0, int(3*ratio*size[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 5)
    # cv2.putText(frame, 'time: ' + str(round(index / 24.0, 2)) + "s", (0,int(4*ratio*size[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 5)
    # frame = cv2.resize(frame, (size*2), interpolation=cv2.INTER_AREA)
    # cv2.imshow("new video", frame)
    cv2.imwrite(path+str(index)+'.jpg',frame)
    # if(index%8==0):
    #     cv2.imwrite(path_out+str(index)+'.png',frame)
    # frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
    # cv2.waitKey(int(1000 /fps)) 
    # videoWriter.write(frame)
    success, frame = video.read()
    index += 1

video.release()