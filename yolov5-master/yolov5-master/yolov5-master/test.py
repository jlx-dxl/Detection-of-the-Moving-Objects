import cv2
import numpy as np
import torch
import time
from utils.plots import Annotator

model = torch.hub.load(repo_or_dir='D:\\for cv\\yolov5-master\\yolov5-master', model='custom',
                                    path=r'D:\for cv\yolov5-master\yolov5-master\yolov5s.pt', source='local')
model.conf = 0.4
#camera = cv2.VideoCapture(0)

cap = cv2.VideoCapture('test.avi')
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
sz = (int(w), int(h))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('\detect result\detect_result.avi', fourcc, fps, sz, isColor=True)

while 1:
    ret, frame = cap.read()
    #frame = cv2.flip(frame, 1)
    if ret == True:
        results = model(frame)
        results_np = results.pandas().xyxy[0].to_numpy()
        #annotator = Annotator(frame, line_width=3, example=str(model.names))
        #frame = annotator.result()

        for box in results_np:
            l, t, r, b = box[:4].astype('int')
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 1)
            cv2.putText(frame, str(box[6]), (l, t-4), 1, 1, (0, 255, 0))
            # 可以在这里做逻辑，黄帽子 显示黄框
        # 显示画面
        cv2.imshow('DEMO', frame)
        out.write(frame)
        # 退出 这种关闭方式哪怕x掉视频框 也会重新打开视频框
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
#camera.release()
cap.release()
out.release()
cv2.destroyAllWindows()
