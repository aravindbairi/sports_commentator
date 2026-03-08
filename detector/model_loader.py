import torch
from torchvision import models


def build_model(num_classes, weights, device=None,checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.video.r2plus1d_18(weights=weights)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=in_features, out_features=num_classes)
    if checkpoint_path:
        print("Loading checkpoint from {}".format(checkpoint_path))
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    model = model.eval().to(device)
    return model, device
