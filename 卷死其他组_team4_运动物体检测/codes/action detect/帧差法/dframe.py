import cv2
import numpy as np
from nms import *
from time import sleep

# video_index = r'D:\for cv\yolov5-master\yolov5-master\test.avi'#摄像头索引或者视频路径
video_index = 0
bias_num = 3#计算帧差图时的帧数差
frame_num = 10#帧率
k_size = 5#中值滤波的滤波器大小
nms_threshold = 0.3#nms阈值
time = 1/frame_num#帧时间
show_test = True#展示二值化结果
threshold = 30#二值化阙值
name = 'frame'#图框名
min_area = 360#目标的最小面积

if not bias_num > 0:
    raise Exception('bias_num must > 0')

if isinstance(video_index, str):
    is_camera = False
else:
    is_camera = True

cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类


frame_num = 0

previous = []

while cap.isOpened():

    catch, frame = cap.read()  # 读取每一帧图片

    if not catch:
        raise Exception('Unexpected Error.')

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

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        status_mask[y:y+h, x:x+w] = 1

    if not is_camera:
        sleep(time)

    cv2.imshow(name, frame)  # 在window上显示图片
    cv2.imshow(name+'status', status_mask)
    if show_test:
        cv2.imshow(name + '_frame', mask)  # 边界

    value = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    previous = pop(previous, value)

    cv2.waitKey(10)

    if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
        # 点x退出
        break

    if show_test and cv2.getWindowProperty(name + '_frame', cv2.WND_PROP_AUTOSIZE) < 1:
        # 点x退出
        break

# 释放摄像头
cap.release()
cv2.destroyAllWindows()



