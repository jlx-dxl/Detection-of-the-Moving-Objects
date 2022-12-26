import copy

import cv2
import numpy as np

class ImageProcessorCloseLoop:
    def __init__(self, subtractor_type="MOG2", tracker_type="DaSiamRPN"):
        self.sub = self.create_subtractor(subtractor_type)
        self.tracker_type = tracker_type
        self.trackers = list()

    def create_subtractor(self, subtractor_type):
        if subtractor_type == "MOG2":
            return cv2.createBackgroundSubtractorMOG2()
        elif subtractor_type == "KNN":
            return cv2.createBackgroundSubtractorKNN()

    def create_tracker(self, tracker_type):
        if tracker_type == "MIL":
            return cv2.TrackerMIL_create()
        elif tracker_type == "GOTURN":
            return cv2.TrackerGOTURN_create()
        elif tracker_type == "DaSiamRPN":
            return cv2.TrackerDaSiamRPN_create()

    def process_one_frame(self, frame):
        '''
        :param frame: original video frame image (np.ndarray)
        :return: bboxes (tuple of bbox)
        '''
        detect_result = self.simple_detect(frame)
        track_result = list()
        for tracker in self.trackers:
            ok, bbox = tracker.update(frame)
            if ok:
                track_result.append(bbox)
            else:
                self.trackers.remove(tracker)
        matched_detect, matched_track, only_detect, only_track = self.compare_result(detect_result, track_result)

        res_matched = list()
        for i in range(len(matched_track)):
            bb1 = matched_detect[i]
            bb2 = matched_track[i]
            bb_res = self.bb_filter2(bb1, bb2)
            res_matched.append(bb_res)

        res_track = only_track
        res_detect = only_detect

        for res in res_detect:
            tracker = self.create_tracker(tracker_type=self.tracker_type)
            # tracker.init(frame, res)

        res = res_matched + res_track + res_detect
        return res

    def simple_detect(self, img: np.ndarray):
        mask = self.sub.apply(img)
        thresh, bw = cv2.threshold(mask, 120, 255, cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), dtype=int)
        dilated = cv2.erode(bw, kernel, iterations=1)
        inflated = cv2.dilate(dilated, kernel, iterations=1)
        # find contour:
        cnts, hier = cv2.findContours(inflated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = list()
        for i in range(len(cnts), 0, -1):
            c = cnts[i - 1]
            area = cv2.contourArea(c)
            if area < 10:
                continue
            bbox = cv2.boundingRect(c)
            x, y, w, h = bbox
            diameter = w if w > h else h
            if diameter <= 15:
                continue
            bboxes.append(bbox)
        return bboxes

    def compare_two(self, bb1, bb2):
        '''
        compare two bbox
        :param bb1: bbox tuple (x, y, w, h)
        :param bb2: bbox tuple
        :return: True or False
        '''
        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2
        IOU1 = (min(x1 + w1, x2 + w2) - max(x1, x2)) / (max(x1 + w1, x2 + w2) - min(x1, x2))
        IOU2 = (min(y1 + h1, y2 + h2) - max(y1, y2)) / (max(y1 + h1, y2 + h2) - min(x1, x2))
        IOU = IOU2 * IOU1
        if IOU > 0.9:
            return True
        else:
            return False

    def compare_result(self, detect_res, track_res):
        '''
        compare the detect results by comparing IOU of the bboxes
        :param last_res: list of bboxes
        :param this_res: list of bboxes
        :return: tuple (matched, new, only_last)
        '''
        matched_detect = list()
        matched_track = list()
        track = list()
        detect = list()

        for track_box in track_res:
            for detect_box in detect_res:
                is_matched = self.compare_two(track_box, detect_box)
                if is_matched:
                    matched_track.append(track_box)
                    matched_detect.append(detect_box)
                    detect_res.remove(detect_box)
                else:
                    track.append(track_box)

        for detect_box in detect_res:
            detect.append(detect_box)
        return matched_detect, matched_track, detect, track

    def bb_filter2(self, bbox1, bbox2, method="mean"):
        '''
        :param bbox1:bbox (x, y, w, h)
        :param bbox2:bbox
        :return: bbox
        '''
        if method == "mean":
            return (bbox1[0] + bbox2[0])/2, (bbox1[1] + bbox2[1])/2, (bbox1[2] + bbox2[2])/2, (bbox1[3] + bbox2[3])/2