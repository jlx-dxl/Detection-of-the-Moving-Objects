import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # 来自vedio视频的

kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
knn = cv2.createBackgroundSubtractorKNN()  # 创建KNN模型

# cap = cv2.VideoCapture(r'D:\for cv\yolov5-master\yolov5-master\test.avi')
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
sz = (int(w), int(h))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(r'D:\for cv\yolov5-master\yolov5-master\detect result\detect_result-knn.avi', fourcc, fps, sz, isColor=True)

while (1):
    # 获取每一帧
    ret, frame = cap.read()

    if frame is None:
        print("camera is over...")
        break

    fmask = knn.apply(frame)  # 判断哪些是前景和背景

    MORPH_OPEN_1 = cv2.morphologyEx(fmask, cv2.MORPH_OPEN, kernel1)  # 开运算，去除噪声和毛刺

    contours, _ = cv2.findContours(fmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 只检测外边框

    for cont in contours:
        # 计算各个轮廓的面积
        len = cv2.arcLength(cont, True)
        if len > 300:  # 去除一些小的噪声点
            # 找到一个轮廓
            x,y,w,h = cv2.boundingRect(cont)
            # 画出这个矩形
            cv2.rectangle(frame, (x,y), (x+w, y+h), color=(0,255,0), thickness=3)

    # 画出所有的轮廓
    cv2.imshow('frame', frame)
    cv2.imshow('fmask', fmask)
    out.write(frame)
    # 进行等待或者退出判断
    if cv2.waitKey(24) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
