import os
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import shutil
import random

class GunshotDataset(Dataset):
    def __init__(self, root_dir, nongunshot_dir, target_sample_rate=16000, n_mels=64):
        self.sample_rate = target_sample_rate
        self.n_mels = n_mels

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels
        )

        self.files = []
        self.labels = []

        # Gunshot label = 1
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            for f in os.listdir(folder_path):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(folder_path, f))
                    self.labels.append(1)

        # Non-gunshot label = 0
        for f in os.listdir(nongunshot_dir):
            if f.endswith(".wav"):
                self.files.append(os.path.join(nongunshot_dir, f))
                self.labels.append(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        waveform, sr = torchaudio.load(filepath)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        mel = self.melspec(waveform)

        # Fix: pad or truncate to fixed length
        max_len = 321  # choose max expected time frames (adjust if needed)
        if mel.shape[2] < max_len:
            pad = max_len - mel.shape[2]
            mel = F.pad(mel, (0, pad))  # pad last dimension
        else:
            mel = mel[:, :, :max_len]

        return mel, label
    
def predict_single_file(model, filepath, sample_rate=16000, n_mels=64, max_len=321, device="cpu"):
    model.eval()  # Set model to evaluation mode

    # Load audio
    waveform, sr = torchaudio.load(filepath)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    # Convert to Mel spectrogram
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels
    )
    mel = melspec(waveform)

    # Pad or truncate to fixed length
    if mel.shape[2] < max_len:
        pad = max_len - mel.shape[2]
        mel = F.pad(mel, (0, pad))
    else:
        mel = mel[:, :, :max_len]

    # Add batch dimension and send to device
    mel = mel.unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(mel)
        pred = output.argmax(dim=1).item()

    return "Gunshot" if pred == 1 else "Non-Gunshot"
    
class GunshotCNN(nn.Module):
    def __init__(self, n_mels=64, time_steps=321):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)

        # Compute flattened size
        x = torch.zeros(1, 1, n_mels, time_steps)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        self.flattened_size = x.numel() // x.shape[0]

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
def prepare_nongunshots(audio_root, dest_folder, max_files=None):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    file_count = 0
    for fold in os.listdir(audio_root):
        fold_path = os.path.join(audio_root, fold)
        if not os.path.isdir(fold_path):
            continue
        for file in os.listdir(fold_path):
            if file.endswith(".wav"):
                src = os.path.join(fold_path, file)
                dst = os.path.join(dest_folder, file)
                shutil.copyfile(src, dst)
                file_count += 1
                if max_files and file_count >= max_files:
                    return
    
def main():
    ## Download latest version of audio dadaset
    pathToGunShot = kagglehub.dataset_download("emrahaydemr/gunshot-audio-dataset")
    print("Path to dataset files:", pathToGunShot)

    # Download UrbanSound8K dataset (non-gunshot noises)
    pathToUrbanSound = kagglehub.dataset_download("chrisfilo/urbansound8k")
    print("UrbanSound dataset at:", pathToUrbanSound)

    # Only run this once
    #prepare_nongunshots(pathToUrbanSound, "Non-Gunshot", max_files=2000)
    #print("Non-Gunshot samples copied.")

    dataset = GunshotDataset(
        root_dir=pathToGunShot,
        nongunshot_dir="Non-Gunshot"
    )

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    #Train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GunshotCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for mel, label in train_loader:
            mel = mel.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(mel)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute batch accuracy
            preds = output.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total * 100

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    model.eval()
    test_file = "testShot.wav"
    result = predict_single_file(model, test_file, device=device)
    print(f"The model predicts for shot: {result}")

    nongunshot_file = "nonShot.wav"
    result2 = predict_single_file(model, nongunshot_file, device=device)
    print(f"The model predicts for non shot: {result2}")

main()