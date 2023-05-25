# python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --img-size 640 --source "dataset/Drone-Detection&Tracking/test/01_2192_0001-1500/000001.jpg"

#python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --img-size 640 --source "dataset/Drone-Detection&Tracking/test/01_2192_0001-1500"
for file in dataset/Drone-Detection\&Tracking/test/*
do
    echo -e "$file"
    python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --img-size 640 --source "$file"
done