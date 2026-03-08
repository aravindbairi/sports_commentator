import os
import random
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

CLASS_NAMES = ["goal","foul"]
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASS_NAMES)}

def read_video_frames(path,num_frames,resize=(112,112)):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames)==num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames)==0:
        #Corrupted file: return black frames
        h,w = resize
        return [np.zeros([h,w,3], dtype=np.uint8) for _ in range(num_frames)]
    #if not enough frames, pad by repeating last frames
    while len(frames)<num_frames:
        frames.append(frames[-1].copy())
    #if more than needed, sample uniformly
    if len(frames)>num_frames:
        idxs = np.linspace(0,len(frames)-1,num=num_frames,dtype=np.int32)
        frames = [frames[i] for i in idxs]
    proc = []
    for frame in frames:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,resize)
        proc.append(frame)
    return proc

class SportsClipDataset(Dataset):
    def __init__(self, root_dir, split="train", T=16, resize=(112,112), transform=None):
        self.T=T
        self.transform = transform
        self.resize = resize
        self.samples = []
        for cname in CLASS_NAMES:
            folder = os.path.join(root_dir,split,cname)
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                if fn.lower().endswith((".mp4",".mov",".mkv",".avi")):
                    self.samples.append((os.path.join(folder,fn),CLASS_TO_IDX[cname]))
        random.shuffle(self.samples)

        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()
                                                    ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path,label = self.samples[idx]
        frames = read_video_frames(path,self.T,self.resize)
        proc = []
        for frame in frames:
            img = self.transform(frame)
            proc.append(img)
        clip = torch.stack(proc,dim=1)
        return clip,torch.tensor(label,dtype=torch.long)
