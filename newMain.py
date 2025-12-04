import os
import random
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# -----------------------------
# Shared Processing Function (FIXED)
# -----------------------------
def get_spectrogram(filepath, target_sample_rate=16000, max_seconds=4.0, **kwargs):
    is_training = kwargs.get('is_training', False)

    # 1. AUDIO LOADING (MISSING SECTION - NOW ADDED)
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception as e:
        # Fallback for corrupt files during training (returns a small, zero tensor)
        # This prevents the DataLoader from crashing the entire run.
        return torch.zeros(1, 64, int(max_seconds * target_sample_rate / 160)) 

    # 2. Resample & Mix to Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)

    # 3. TEMPORAL DATA AUGMENTATION (TIME SHIFT)
    if is_training:
        max_shift = int(target_sample_rate * 0.5)  # Shift up to 0.5 seconds
        if waveform.shape[1] > max_shift: # Only shift if clip is long enough
            shift_amount = random.randint(-max_shift, max_shift)
            
            # Apply padding for the shift amount, then crop the shifted part
            waveform = F.pad(waveform, (abs(shift_amount), 0), mode='constant', value=0.)[:, :waveform.shape[1]]
            waveform = waveform[:, max(0, -shift_amount):waveform.shape[1] - max(0, shift_amount)]


    # 4. Create Mel Spectrogram
    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_mels=64,
        n_fft=400,
        hop_length=160
    )
    mel = melspec_transform(waveform)

    # 5. Convert to Decibels & Normalization
    db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
    mel = db_transform(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    
    # 6. Pad or Crop (Handling different clip lengths)
    max_len = int(max_seconds * target_sample_rate / 160)
    current_len = mel.shape[2]

    if current_len < max_len:
        # Pad with a value close to the mean/silence
        pad_amount = max_len - current_len
        mel = F.pad(mel, (0, pad_amount), mode='constant', value=mel.mean()) 
    elif current_len > max_len and is_training:
        # Random Crop for longer files during training
        start = random.randint(0, current_len - max_len)
        mel = mel[:, :, start:start + max_len]
    else:
        # For evaluation, take the first segment
        mel = mel[:, :, :max_len] 

    return mel

# -----------------------------
# Dataset
# -----------------------------
class UrbanSoundGunshotDataset(Dataset):
    def __init__(self, root_dir, use_folds=None, use_additional_gunshots=False, additional_gunshot_dir=None):
        self.files = []
        self.labels = []

        # Load UrbanSound8K
        metadata_path = os.path.join(root_dir, "UrbanSound8K.csv")
        metadata = pd.read_csv(metadata_path)

        for _, row in metadata.iterrows():
            fold = row['fold']
            class_name = row['class']
            if use_folds and fold not in use_folds:
                continue
            
            filepath = os.path.join(root_dir, f"fold{fold}", row['slice_file_name'])
            self.files.append(filepath)
            self.labels.append(1 if class_name == "gun_shot" else 0)

        # Load Additional Gunshots
        if use_additional_gunshots and additional_gunshot_dir:
            for root, _, filenames in os.walk(additional_gunshot_dir):
                for fname in filenames:
                    if fname.endswith(".wav"):
                        fpath = os.path.join(root, fname)
                        self.files.append(fpath)
                        self.labels.append(1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        mel = get_spectrogram(filepath, is_training=True)
        return mel, label

# -----------------------------
# Improved Model (With Batch Norm)
# -----------------------------
class GunshotCNN(nn.Module):
    def __init__(self, n_mels=64, max_len=None):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # Batch Norm helps convergence
        self.pool1 = nn.MaxPool2d(2)

        # Block 2
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        # Block 3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.4)

        # Calculate flattened size dynamically
        if max_len is None:
            max_len = 400 
        
        # Dummy pass to get size
        with torch.no_grad():
            x = torch.zeros(1, 1, n_mels, max_len)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            self.flattened_size = x.numel()

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# -----------------------------
# Predictor
# -----------------------------
def predict_file(model, filepath, device="cpu"):
    model.eval()
    mel = get_spectrogram(filepath)
    mel = mel.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(mel)
        # Apply Softmax to get percentages
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
    
    print(f"{os.path.basename(filepath)} - Prob: Gun={probs[1]:.4f}, Non={probs[0]:.4f}")

# -----------------------------
# Main
# -----------------------------
def main():
    path_to_urbansound = kagglehub.dataset_download("chrisfilo/urbansound8k")
    path_to_gunshots = kagglehub.dataset_download("emrahaydemr/gunshot-audio-dataset")

    # 1. Setup Dataset
    train_dataset = UrbanSoundGunshotDataset(
        root_dir=path_to_urbansound,
        use_folds=list(range(1,10)),
        use_additional_gunshots=True,
        additional_gunshot_dir=path_to_gunshots
    )

    # 2. Balance the Data (50% Gunshot, 50% Non-Gunshot)
    gunshot_indices = [i for i, lbl in enumerate(train_dataset.labels) if lbl == 1]
    nongunshot_indices = [i for i, lbl in enumerate(train_dataset.labels) if lbl == 0]

    random.seed(42)
    # Take only as many non-gunshots as we have gunshots
    sampled_nongunshot = random.sample(nongunshot_indices, len(gunshot_indices))
    
    balanced_indices = gunshot_indices + sampled_nongunshot
    random.shuffle(balanced_indices)
    
    balanced_train_dataset = Subset(train_dataset, balanced_indices)
    train_loader = DataLoader(balanced_train_dataset, batch_size=32, shuffle=True)

    # 3. Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = int(4.0 * 16000 / 160)
    model = GunshotCNN(n_mels=64, max_len=max_len).to(device)
    
    # 4. Optimizer & Loss 
    # REMOVED WEIGHTS: Data is already balanced!
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss() 

    # 5. Train
    epochs = 10
    print(f"Training on {len(balanced_train_dataset)} balanced samples...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for mel, label in train_loader:
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(mel)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (output.argmax(1) == label).sum().item()
            total += label.size(0)
        
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}, Acc {correct/total*100:.2f}%")

    # 6. Test
    print("\n--- Predictions ---")
    model.eval()
    
    test_file = "testShot.wav"
    if os.path.exists(test_file): predict_file(model, test_file, device=device)

    non_file = "nonShot.wav"
    if os.path.exists(non_file): predict_file(model, non_file, device=device)

    print("\n--- Pistol Folder ---")
    if os.path.exists("shots/pistol"):
        for file in os.listdir("shots/pistol"):
            if file.endswith(".wav"):
                predict_file(model, os.path.join("shots/pistol", file), device=device)
    
    print("\n--- Long Pistol Folder ---")
    if os.path.exists("shots/pistolLonger"):
        for file in os.listdir("shots/pistolLonger"):
            if file.endswith(".wav"):
                predict_file(model, os.path.join("shots/pistolLonger", file), device=device)

    # Model Saving Logic
    model_save_path = "gunshot_cnn_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n Model saved  to: {model_save_path}")

main()