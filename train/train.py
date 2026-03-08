import argparse
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import models

from dataset.dataset import SportsClipDataset, CLASS_NAMES


def build_model(num_classes,device,pretrained=True):
    model = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features,out_features=num_classes)
    model = model.to(device)
    return model

def train_model(model,loader,optim,loss_fn,device,scaler=None):
    model.train()
    total_loss = 0
    n=0
    for clips,labels in loader:
        clips = clips.to(device,non_blocking=True)
        labels = labels.to(device,non_blocking=True)
        optim.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(clips)
                loss = loss_fn(logits,labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits = model(clips)
            loss = loss_fn(logits,labels)
            loss.backward()
            optim.step()
        total_loss = total_loss+loss.item()*clips.size(0)
        n+=clips.size(0)
    return total_loss/n

def validate_model(model,loader,device,scaler=None):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for clips,labels in loader:
            clips = clips.to(device,non_blocking=True)
            logits = model(clips)
            preds = torch.argmax(logits,dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    p,r,f,_ = precision_recall_fscore_support(y_true,y_pred,labels=list(range(len(CLASS_NAMES))),zero_division=0,average=None)
    return p,r,f,y_true,y_pred

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    train_dataset = SportsClipDataset(args.data_root,"train",T=args.T,resize=(args.sz,args.sz))
    val_ds = SportsClipDataset(args.data_root,"test",resize=(args.sz,args.sz))
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=torch.cuda.is_available(),num_workers=2)
    val_loader = DataLoader(val_ds,batch_size=args.batch_size,shuffle=True,pin_memory=torch.cuda.is_available(),num_workers=2)
    model = build_model(num_classes=len(CLASS_NAMES),device=device,pretrained=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if device=="cuda" and args.use_amp else None
    best_f1 = 0
    for epoch in range(1,args.epochs+1):
        t0 = time.time()
        train_loss = train_model(model,train_loader,optimizer,loss_fn,device,scaler)
        p,r,f,y_true,y_pred = validate_model(model,val_loader,device,scaler)
        mean_f1 = float(f.mean())
        t1 = time.time()
        print("Epoch:",epoch,"loss",train_loss,"time",t1-t0,"mean_f1",mean_f1)
        for i, cname in enumerate(CLASS_NAMES):
            print(f"class:{cname} Precision {p[i]:.3f} Recall {r[i]:.3f} F1 {f[i]:.3f}")
            ckpt = os.path.join(args.save_dir,f"model_epoch{epoch}.pth")
            torch.save(model.state_dict(),ckpt)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                torch.save(model.state_dict(),os.path.join(args.save_dir,"best_model.pth"))
                print("best_model.pth:",best_f1)
    print("Training finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",type=str,default="./data")
    parser.add_argument("--save_dir",type=str,default="./checkpoints")
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--sz",type=int,default=112)
    parser.add_argument("--T",type=int,default=16)
    # parser.add_argument("use_amp",action="store_true")
    args = parser.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)
    main(args)




