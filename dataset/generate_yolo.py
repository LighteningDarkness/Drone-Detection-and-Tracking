import os

if __name__=="__main__":
    root="dataset/Drone-Detection&Tracking"
    train_root="dataset/Drone-Detection&Tracking/train"
    test_root="dataset/Drone-Detection&Tracking/test"

    videos=os.listdir(train_root)
    point=int(len(videos)*0.8)

    train_data=[]
    for i,v in enumerate(videos):
        if i<point:
            images=os.listdir(os.path.join(train_root,v))
            for img in images:
                if img.endswith(".jpg"):
                    train_data.append(os.path.join(train_root,v,img))

    valid_data=[]
    for i,v in enumerate(videos):
        if i>=point:
            images=os.listdir(os.path.join(train_root,v))
            for img in images:
                if img.endswith(".jpg"):
                    valid_data.append(os.path.join(train_root,v,img))

    videos=os.listdir(test_root)

    test_data=[]
    for i,v in enumerate(videos):
        images=os.listdir(os.path.join(test_root,v))
        for img in images:
            if img.endswith(".jpg"):
                test_data.append(os.path.join(test_root,v,img))
    train_data=sorted(train_data)
    valid_data=sorted(valid_data)
    test_data=sorted(test_data)
    with open("dataset/train_list.txt","w+",encoding='utf-8') as f:
        for i in train_data:
            f.writelines(i+"\n")
    
    with open("dataset/valid_list.txt","w+",encoding='utf-8') as f:
        for i in valid_data:
            f.writelines(i+"\n")
    with open("dataset/test_list.txt","w+",encoding='utf-8') as f:
        for i in test_data:
            f.writelines(i+"\n")
