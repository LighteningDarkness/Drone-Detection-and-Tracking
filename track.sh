for file in dataset/Drone-Detection\&Tracking/test/*
do
    echo -e "$file"
    python track_demo.py --source "$file"
done