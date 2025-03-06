import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class Audio_Dataset(Dataset):
    def __init__(self, df, target_size=(128, 96)):
        self.labels = df["label"].values
        self.image_arrays = df["audio_arrays"].values
        self.target_size = target_size  # Fixed size for all tensors

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = torch.tensor(self.image_arrays[index], dtype=torch.float32)

        # Ensure image has a single channel (for CNN)
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Shape: (1, 128, 96)

        # Resize to target size (128, 96)
        image = F.interpolate(image.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label
