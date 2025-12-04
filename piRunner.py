import torch
import torch.nn.functional as F
import torchaudio
import torch.nn as nn
import numpy as np
import pyaudio
import time
import random

# --- CONFIGURATION (MUST MATCH TRAINING) ---
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.0 # Process 1-second chunks (matches your working pistol length)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_SECONDS)
CHANNELS = 1
FORMAT = pyaudio.paInt16

MODEL_PATH = "gunshot_cnn_model.pth"
MAX_SECONDS = 4 # The fixed length your model expects
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160
DEVICE = "cpu" # Raspberry Pi deployment

# --- MODEL DEFINITION (MUST MATCH TRAINING) ---
# Paste your GunshotCNN class definition here exactly as it was during training
class GunshotCNN(nn.Module):
    def __init__(self, n_mels=64, max_len=None):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        # Block 2
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        # Block 3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.5) # Assuming you set this to 0.5

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


# --- PROCESSING FUNCTION (Adapted for Real-time Stream) ---
def process_chunk(audio_data):
    """Converts raw audio chunk (numpy array) to a normalized Mel-spectrogram tensor."""
    
    # 1. Convert raw bytes to torch tensor
    # PaInt16 to float32 normalized by max int value (32768)
    waveform = torch.from_numpy(audio_data.astype(np.float32) / 32768.0).unsqueeze(0)
    
    # 2. Resample (Not needed if mic is set to 16k, but good practice if needed)
    # The microphone settings should be configured to capture at 16000 Hz.
    
    # 3. Create Mel Spectrogram
    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel = melspec_transform(waveform)

    # 4. Convert to Decibels & Normalization (Crucial!)
    db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
    mel = db_transform(mel)
    
    # Standardization (Mean/Std)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    
    # 5. Padding/Cropping to MAX_SECONDS (4.0s) length
    # This section ensures the 1-second chunk is correctly placed in the 4-second input your model expects.
    max_len = int(MAX_SECONDS * SAMPLE_RATE / HOP_LENGTH)
    
    if mel.shape[2] < max_len:
        # Pad the 1-second clip to fill the 4-second model input
        pad_amount = max_len - mel.shape[2]
        # Pad with a value close to the mean/silence
        mel = F.pad(mel, (0, pad_amount), mode='constant', value=mel.mean())
    else:
        # Should not happen if chunk is 1 second, but ensures size is correct
        mel = mel[:, :, :max_len] 

    return mel.unsqueeze(0).to(DEVICE) # Add batch dimension

# --- MAIN DEPLOYMENT LOGIC ---
def run_detector():
    """Initializes model, PyAudio stream, and runs the continuous loop."""
    
    # 1. Load Model
    max_len = int(MAX_SECONDS * SAMPLE_RATE / HOP_LENGTH)
    model = GunshotCNN(n_mels=N_MELS, max_len=max_len).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Exiting.")
        return
    
    # 2. Setup PyAudio Stream
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("Starting audio stream. Listening for gunshots...")

    # 3. Continuous Detection Loop
    try:
        while True:
            # Read a chunk of audio data
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # Convert raw bytes to a NumPy array (int16)
            np_data = np.frombuffer(data, dtype=np.int16)

            # Process and convert to model input tensor
            input_tensor = process_chunk(np_data)
            
            # Run prediction
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                gun_prob = probs[1]
            
            # 4. Reporting
            if gun_prob > 0.60: # Set a confident threshold (e.g., 90%)
                print(f"GUNSHOT DETECTED! Probability: {gun_prob:.4f} @ {time.strftime('%H:%M:%S')}")
            elif gun_prob > 0.45:
                print(f" High confidence event: {gun_prob:.4f}")
            else:
                 # Print periodically to show the script is still running
                 if int(time.time()) % 10 == 0: 
                    print(f"Listening... (Gun Prob: {gun_prob:.4f})")
            
            time.sleep(CHUNK_SECONDS / 2) # Wait slightly less than CHUNK_SECONDS for continuous listening

    except KeyboardInterrupt:
        print("\nStopping detector...")
    finally:
        # 5. Clean Up
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    run_detector()