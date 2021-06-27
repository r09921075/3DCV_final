import cv2
import glob

path = "data/need_img/car/*.png"     #input frame
result_name = 'data/video_out/car/car_4.mp4'

frame_list = sorted(glob.glob(path),key=len)
print("frame count: ",len(frame_list))
ratio = 0.1
fps = 4  #fps
shape = cv2.imread(frame_list[0]).shape # delete dimension 3
size = (shape[1], shape[0])
print("frame size: ",size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(result_name, fourcc, fps, size)

for idx, path in enumerate(frame_list):
    frame = cv2.imread(path)
    # print("\rMaking videos: {}/{}".format(idx+1, len(frame_list)), end = "")
    current_frame = idx+1
    total_frame_count = len(frame_list)
    # cv2.putText(frame, 'fps: ' + str(fps), (0,int(1*ratio*size[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)
    # cv2.putText(frame, 'frame: ' + str(idx), (0, int(2*ratio*size[1])), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,255), 3)
    # cv2.putText(frame, 'frame: ' + str(idx), (0, int(3*ratio*size[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 5)
    percentage = int(current_frame*30 / (total_frame_count+1))
    print("\rProcess: [{}{}] {:06d} / {:06d}".format("#"*percentage, "."*(30-1-percentage), current_frame, total_frame_count), end ='')
    out.write(frame)

out.release()
print("Finish making video !!!")


