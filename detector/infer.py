import time

import cv2
import numpy as np
import torch
from torchvision import transforms,models

from dataset.dataset import CLASS_NAMES
from detector.model_loader import build_model

SPATIAL_SIZE = 112
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((SPATIAL_SIZE, SPATIAL_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

def frames_to_tensor(frames):
    processed_frames = []
    for frame in frames:
        f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(transform(f_rgb))
    clip = torch.stack(processed_frames, dim=1)
    clip = clip.unsqueeze(0)
    return clip

class Detector:
    def __init__(self, checkpoint, device=None,num_classes=len(CLASS_NAMES), weights=models.video.R2Plus1D_18_Weights.DEFAULT):
        self.model,self.device = build_model(num_classes=num_classes, weights=weights, device=device,checkpoint_path=checkpoint)
        self.model.eval()
        self.class_names = CLASS_NAMES[:num_classes]

    @torch.no_grad()
    def infer_clip(self,frames,topk=3):
        tensor = frames_to_tensor(frames).to(self.device)
        start_time = time.time()
        logits = self.model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
        end_time = time.time()
        idxs = np.argsort(probs)[::-1][:topk]
        results = [(self.class_names[int(i)], float(probs[int(i)])) for i in idxs]
        return {"topk": results, "time": (end_time - start_time)*1000, "raw_probs": probs}

    def infer_batch(self,batch_frames,topk=3):
        return [self.infer_clip(frame,topk) for frame in batch_frames]