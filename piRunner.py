import torch
import torch.nn.functional as F
import torchaudio
import torch.nn as nn
import numpy as np
import pyaudio
import time
import argparse
import discord
import asyncio

# --- CONFIGURATION (MUST MATCH TRAINING) ---
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.0 
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_SECONDS)
CHANNELS = 1
FORMAT = pyaudio.paInt16

MODEL_PATH = "gunshot_cnn_model.pth"
MAX_SECONDS = 4 
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160
DEVICE = "cpu" 

# --- GLOBAL NORMALIZATION STATS (ENSURE THESE MATCH YOUR SAVED MODEL) ---
GLOBAL_MEAN = -13.3991 
GLOBAL_STD = 21.2402 
DISCORD_CHANNEL_ID = 1446077899835707493

# --- MODEL DEFINITION (Must match your saved model architecture) ---
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

        self.dropout = nn.Dropout(0.5) 

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


# --- PROCESSING FUNCTION (Using Global Stats) ---
def process_chunk(audio_data):
    """Converts raw audio chunk (numpy array) to a normalized Mel-spectrogram tensor."""
    
    waveform = torch.from_numpy(audio_data.astype(np.float32) / 32768.0).unsqueeze(0)
    
    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel = melspec_transform(waveform)

    db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
    mel = db_transform(mel)
    
    # Use Global Standardization
    mel = F.pad(mel, (0, pad_amount), mode='constant', value=mel.mean())
    
    # Padding/Cropping to MAX_SECONDS (4.0s) length
    max_len = int(MAX_SECONDS * SAMPLE_RATE / HOP_LENGTH)
    
    if mel.shape[2] < max_len:
        pad_amount = max_len - mel.shape[2]
        # Pad with the GLOBAL mean/silence value
        mel = F.pad(mel, (0, pad_amount), mode='constant', value=GLOBAL_MEAN) 
    else:
        mel = mel[:, :, :max_len] 

    return mel.unsqueeze(0).to(DEVICE)

# --- DISCORD SENDER FUNCTION ---
async def send_discord_message(client, message):
    """Asynchronously sends a message to the specified Discord channel."""
    await client.wait_until_ready()
    try:
        channel = client.get_channel(DISCORD_CHANNEL_ID)
        if channel:
            await channel.send(message)
        else:
            print(f"Error: Discord channel with ID {DISCORD_CHANNEL_ID} not found.")
    except Exception as e:
        print(f"Error sending Discord message: {e}")
    await asyncio.sleep(0.1) # Allow other async tasks to run


# --- ASYNC DETECTION LOOP ---
async def detection_loop(model, stream, p, discord_client):
    last_detection_time = 0
    COOLDOWN = 10 
    
    print("Starting audio stream detection...")

    try:
        while True:
            # Blocking audio read (must be fast enough not to stall the loop too long)
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            np_data = np.frombuffer(data, dtype=np.int16)
            input_tensor = process_chunk(np_data)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                gun_prob = probs[1]
            
            current_time = time.time()
            
            # 4. Reporting
            if gun_prob > 0.75:
                print(f"🚨 GUNSHOT DETECTED! Probability: {gun_prob:.4f} @ {time.strftime('%H:%M:%S')}")
                if (current_time - last_detection_time) > COOLDOWN:
                    discord_message = f"**ALERT!** Gunshot detected @ {time.strftime('%Y-%m-%d %H:%M:%S')}. Prob: {gun_prob:.2f}"
                    discord_client.loop.create_task(send_discord_message(discord_client, discord_message))
                    last_detection_time = current_time

            elif gun_prob > 0.6:
                print(f" High confidence event: {gun_prob:.4f}")
            else:
                if int(time.time()) % 10 == 0: 
                    print(f"Listening... (Gun Prob: {gun_prob:.4f})")
            
            await asyncio.sleep(CHUNK_SECONDS / 2) # Use asyncio.sleep

    except KeyboardInterrupt:
        print("\nStopping detector...")
    finally:
        # Clean Up
        stream.stop_stream()
        stream.close()
        p.terminate()
        await discord_client.close()


# --- MAIN DEPLOYMENT LOGIC ---
def run_detector(token):
    
    # 1. Load Model
    max_len = int(MAX_SECONDS * SAMPLE_RATE / HOP_LENGTH)
    model = GunshotCNN(n_mels=N_MELS, max_len=max_len).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error: Model file not found or corrupted. Error: {e}. Exiting.")
        return
    
    # 2. Setup PyAudio Stream
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("Starting audio stream. Waiting for Discord bot to connect...")

    # 3. Setup Discord Bot
    intents = discord.Intents.default()
    intents.message_content = True 
    discord_client = discord.Client(intents=intents)

    @discord_client.event
    async def on_ready():
        print(f'Discord bot logged in as {discord_client.user}')
        # Start the detection loop as an async task
        discord_client.loop.create_task(detection_loop(model, stream, p, discord_client))

    # Run the Discord bot
    try:
        # discord_client.run(token) is blocking and keeps the program alive
        discord_client.run(token)
    except Exception as e:
        print(f"Discord Bot Error: Ensure your token is correct. Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Raspberry Pi Gunshot Detector with Discord Notifications.")
    parser.add_argument("--token", required=True, help="Your Discord bot token.")
    args = parser.parse_args()
    
    run_detector(args.token)