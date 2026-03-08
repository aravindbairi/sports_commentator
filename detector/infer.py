import cv2
import torch
from torchvision import models, transforms

from dataset.dataset import CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.video.r2plus1d_18(pretrained=False)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features=in_features,out_features=len(CLASS_NAMES))
model.load_state_dict(torch.load("../checkpoints/best_model.pth", map_location=device))
model = model.eval().to(device)

spatial_size = 112
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((spatial_size, spatial_size)),
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

@torch.no_grad()
def infer_clip(frames,topk=3):
    tensor = frames_to_tensor(frames).to(device)
    logits = model(tensor)
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    topk_values, topk_indices = torch.topk(probs, k=topk)
    return [(int(idx.item()),float(val.item())) for idx,val in zip(topk_indices,topk_values)]

def detect_event(frame_number):
    if frame_number==200:
        return {
            "event": "goal",
            "team": "Portugal",
            "player": "Ronaldo",
            "time": "00:07"
        }
    return None