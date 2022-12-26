import cv2
import processor

if __name__=="__main__":
    cap = cv2.VideoCapture("/Users/wanglingyu/sources/video1.mp4")

    proc = processor.ImageProcessorCloseLoop(tracker_type="MIL")

    # ret, frame = cap.read()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes = proc.process_one_frame(frame)
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0xff), 1)

        cv2.imshow("Video", frame)
        key = cv2.waitKey(10)
        if key == ord(" "):
            k1 = cv2.waitKey()


    cap.release()
    cv2.destroyAllWindows()