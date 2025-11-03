# train_posenet.py

import os
import csv
import numpy as np
from PIL import Image
from trimesh.transformations import quaternion_from_euler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from network import PoseNet

class PoseDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        #prepare data dir for traning
        self.images_dir = os.path.join(dataset_dir, "images")
        poses_file = os.path.join(dataset_dir, "poses.csv")
        self.transform = transform
        self.samples = []
        with open(poses_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx     = int(row["image_number"])
                img_fn  = os.path.join(self.images_dir, f"{idx:04d}.png")
                tx, ty, tz   = float(row["tx"]), float(row["ty"]), float(row["tz"])
                roll, pitch, yaw = float(row["rx"]), float(row["ry"]), float(row["rz"])
                # convert to quaternion [x, y, z, w]
                quat = quaternion_from_euler(roll, pitch, yaw, axes="sxyz")
                # enforce a unique hemisphere
                if quat[3] < 0:
                    quat = -quat
                t = np.array([tx, ty, tz], dtype=np.float32)
                q = np.array(quat,      dtype=np.float32)
                self.samples.append((img_fn, t, q))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_fn, t, q = self.samples[idx]
        img = Image.open(img_fn).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = np.concatenate([t, q], axis=0)
        return img, torch.from_numpy(target)

def train():
    # hyperparameters
    dataset_dir = "dataset"
    epochs = 200
    batch_size = 32
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # normalization transform for resnet18 imagenet weights, constants found from source cited on report
    transform = T.Compose([T.Pad((0, 80, 0, 80)),T.Resize((224, 224)),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    dataset = PoseDataset(dataset_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = PoseNet(num_outputs=7).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "pose_net_2.pth")
    print("Model saved")

if __name__ == "__main__":
    train()
