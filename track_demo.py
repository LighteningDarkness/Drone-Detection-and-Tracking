import argparse
import os
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized, TracedModel
import time 
from ByteTrack.tracker.byte_tracker import BYTETracker
from utils.visualize import plot_tracking
from ByteTrack.tracking_utils.timer import Timer
from inference import Detect
import json
def track_demo(video_path="dataset/Drone-Detection&Tracking/test/01_2192_0001-1500",save_txt=True):
    txt_dir = "result_bt"
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    
    # Detected
    conf_thres = 0.1
    iou_thres = 0.25
    img_size = 640
    weights = "runs/train/exp/weights/best.pt"
    device = 0
    half_precision = True
    deteted = Detect(weights, device, img_size, conf_thres, iou_thres, single_cls=False, half_precision=half_precision, trace= False)
   
    # Tracking
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    frame_rate = 25
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20_check = False
    res_file = os.path.join(txt_dir, video_path.split('/')[-1]+".txt")

    print(track_thresh, track_buffer, match_thresh, mot20_check, frame_rate)
    tracker = BYTETracker(track_thresh, track_buffer, match_thresh, mot20_check, frame_rate)
    timer = Timer()
    # cap = cv2.VideoCapture(video_path)
    frames=sorted(os.listdir(video_path))
    frames.remove("IR_label.json")
    frame_id = 0
    results = []
    for i,frame in enumerate(frames):
        im0=cv2.imread(os.path.join(video_path,frame))
        height, width, _ = im0.shape
        t1 = time.time()
        #2-dim list
        if i==0:
            with open(os.path.join(video_path,"IR_label.json")) as f:
                res_first=json.load(f)
            dets=res_first["res"] if "res" in res_first.keys() else res_first["gt_rect"][0:1]
            dets[0][2]+=dets[0][0]
            dets[0][3]+=dets[0][1]
            dets[0].append(1.0)
        else:
            dets = deteted.inference(im0)
        if len(dets)>1:
            max_idx=0
            for i,d in enumerate(dets):
                if d[4]>dets[max_idx][4]:
                    max_idx=i
            dets=dets[max_idx:max_idx+1]
        online_targets = tracker.update(np.array(dets), [height, width], (height, width)) if len(dets)!=0 else []
        online_tlwhs = []
        online_ids = []
        online_scores = []
        #print(len(online_targets)==0)
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            # save result for evaluation
            tmp_list=[int(tlwh[0]),int(tlwh[1]),int(tlwh[2]),int(tlwh[3])]
            results.append(
                tmp_list
            )

        if online_targets==[]:
            results.append([])
        t2 = time.time()
        print(f"FPS:{1 /(t2-t1):.2f}")
        timer.toc()
        #print(1. / timer.average_time)
        #online_im = plot_tracking(im0, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / 1 /(t2-t1))
    if save_txt:
        with open(res_file, 'w+') as f:
            f.write(str({"res":results}))
        #cv2.imshow("Frame", online_im)
    return results
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    opt = parser.parse_args()
    track_demo(opt.source)