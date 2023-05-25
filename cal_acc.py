from track_demo import track_demo
import os
import json
from inference import Detect
import cv2
import argparse
def cal_iou(b1,b2):
    x1,y1,w1,h1=b1
    x2,y2,w2,h2=b2
    a=min(x1+w1,x2+w2)-max(x1,x2) if min(x1+w1,x2+w2)>max(x1,x2) else 0
    b=min(y1+h1,y2+h2)-max(y1,y2) if min(y1+h1,y2+h2)>max(y1,y2) else 0
    #print(a*b/(w1*h1+w2*h2-a*b))
    return a*b/(w1*h1+w2*h2-a*b)
def cal_acc(args):
    with open("dataset/valid_list.txt") as f:
        valid_set=f.readlines()
    valid_video=set()
    for i in valid_set:
        tmp=i.split('/')
        valid_video.add(tmp[-2])
    acc=dict()

    for v in valid_video:
        TP=0
        TN=0
        FN=0
        gt_seen=0
        if args.method=="ByteTrack":
            results=track_demo(os.path.join("dataset/Drone-Detection&Tracking/train",v),save_txt=False)
        else:
            results=[]
            model=Detect()
            imgs=os.listdir(os.path.join("dataset/Drone-Detection&Tracking/train",v))
            for im in sorted(imgs):
                if im.endswith(".jpg"):
                    im0=cv2.imread(os.path.join("dataset/Drone-Detection&Tracking/train",v,im))
                    dets=model.inference(im0)
                    if len(dets)>1:
                        max_idx=0
                        for i,d in enumerate(dets):
                            if d[4]>dets[max_idx][4]:
                                max_idx=i
                        dets=dets[max_idx:max_idx+1]
                    if len(dets)>0:
                        dets[0]=dets[0][0:4]
                        dets[0][2]-=dets[0][0]
                        dets[0][3]-=dets[0][1]
                    det=dets[0] if len(dets)>0 else []
                    results.append(det)
        with open(os.path.join("dataset/Drone-Detection&Tracking/train",v,"IR_label.json")) as f:
            gt=json.load(f)
        print(v)
        assert len(results)==len(gt["exist"])
        for i in range(len(results)):
            if results[i]!=[] and gt["exist"][i]!=0:
                iou=cal_iou(results[i],gt["gt_rect"][i])
                # print(iou)
                TP+=iou
                gt_seen+=1
            elif results[i]==[] and gt["exist"][i]==0:
                TN+=1
            elif results[i]==[] and gt["exist"][i]!=0:
                gt_seen+=1
                FN+=1
        acc[v]=(TP+TN)/len(results)-0.2*(FN/gt_seen)**0.3
        print(f"acc of {v}:{acc[v]}")
    print(acc)
    print(f"avg acc:{sum(acc.values(),0)/len(acc.keys())}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ByteTrack', help='source')  # file/folder, 0 for webcam
    opt = parser.parse_args()
    cal_acc(opt)