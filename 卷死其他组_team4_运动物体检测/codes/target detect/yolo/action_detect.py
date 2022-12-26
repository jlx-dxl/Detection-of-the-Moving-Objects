import cv2
import numpy as np
from nms import *
from time import sleep
import torch

## parameter define
video_index = 0#摄像头索引或者视频路径
bias_num = 3#计算帧差图时的帧数差
frame_num = 10#帧率
k_size = 5#中值滤波的滤波器大小
nms_threshold = 0.3#nms阈值
show_test = True#展示二值化结果
threshold = 30#二值化阙值
name = 'frame'#图框名q
min_area = 20#目标的最小面积
time = 1/frame_num#一帧时间
act_threshold = 0.1

model = torch.hub.load(repo_or_dir='', model='custom',
                                    path='yolov5s.pt', source='local')
model.conf = 0.4

if isinstance(video_index, str):
    is_camera = False
else:
    is_camera = True

cap = cv2.VideoCapture(video_index)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
sz = (int(w), int(h))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(r'D:\for cv\yolov5-master\yolov5-master\detect result\action_detect_result_cellphone.avi', fourcc, fps, sz, isColor=True)
frame_num = 0
previous = []
pre_loc = []
now_loc = []
v_threshold = 50
fcnt = 0
dif = 5
while 1:
    ret, frame = cap.read()
    fcnt += 1
    frame = cv2.flip(frame, 1)
    if ret == True:
        results = model(frame)
        results_np = results.pandas().xyxy[0].to_numpy()
        #annotator = Annotator(frame, line_width=3, example=str(model.names))
        #frame = annotator.result()

        if frame_num < bias_num:
            value = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            previous.append(value)

            frame_num += 1

        raw = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.absdiff(gray, previous[0])
        gray = cv2.medianBlur(gray, k_size)

        ret, mask = cv2.threshold(
            gray, threshold, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounds = nms_cnts(cnts, mask, min_area, nms_threshold)

        status_mask = np.zeros(mask.shape)

        for b in bounds:
            x, y, w, h = b
            status_mask[y:y + h, x:x + w] = 1

        for box in results_np:
            v = -1
            l, t, r, b = box[:4].astype('int')
            rect_loc = [(l+r)/2, (t+b)/2]
            now_loc.append(rect_loc)
            area = (r-l)*(b-t)
            act_area = sum(sum(status_mask[t:b,l:r]))
            if len(pre_loc) != 0:
                diss = [np.sqrt((i[0]-rect_loc[0])**2+(i[1]-rect_loc[1])**2) for i in pre_loc]
                if min(diss) < v_threshold:
                    v = round(min(diss),2)
            if str(box[6])!='bird':
                if act_area/area>act_threshold:
                    if v>0:
                        cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 1)
                        cv2.putText(frame, str(box[6])+' action', (l, t-4), 1, 1, (0, 0, 255))
                        cv2.putText(frame, str(v)+'pts/f', (l, b + 10), 1, 1, (0, 0, 255))
                    else:
                        cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 1)
                        cv2.putText(frame, str(box[6]) + ' action', (l, t - 4), 1, 1,
                                    (0, 0, 255))
                else:
                    cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 1)
                    cv2.putText(frame, str(box[6])+' static', (l, t - 4), 1, 1, (0, 255, 0))
                # 可以在这里做逻辑，黄帽子 显示黄框
        # 显示画面
#        if not is_camera:
#            sleep(0.05)
        pre_loc = now_loc
        now_loc = []

        cv2.imshow('DEMO', frame)
#        out.write(frame)

        value = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        previous = pop(previous, value)

        # 退出 这种关闭方式哪怕x掉视频框 也会重新打开视频框
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
#camera.release()
cap.release()
#out.release()
cv2.destroyAllWindows()