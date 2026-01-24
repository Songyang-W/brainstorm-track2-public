import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
from scipy.signal import butter, sosfilt

# ==========================================
# 1. LIVE SIGNAL PROCESSOR
# ==========================================
class LiveProcessor:
    def __init__(self, fs=500, n_channels=1024):
        self.fs = fs
        self.sos = butter(4, [70, 150], btype='band', fs=fs, output='sos')
        n_sections = self.sos.shape[0]
        self.zi = np.zeros((n_sections, 2, n_channels))
        self.prev_smooth = None
        self.alpha = 0.2 

    def process_packet(self, raw_packet):
        filtered, self.zi = sosfilt(self.sos, raw_packet, axis=0, zi=self.zi)
        power = filtered ** 2
        avg_power = np.mean(power, axis=0)
        grid = avg_power.reshape(32, 32)
        
        if self.prev_smooth is None:
            self.prev_smooth = grid
        else:
            self.prev_smooth = self.alpha * grid + (1 - self.alpha) * self.prev_smooth
            
        return self.prev_smooth

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class MedicalUNet(nn.Module):
    def __init__(self):
        super(MedicalUNet, self).__init__()
        self.e1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)
        self.e2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU())
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU()) 
        self.upconv2 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.d2 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.e1(x)
        p1 = self.pool1(x1)
        x2 = self.e2(p1)
        p2 = self.pool2(x2)
        b = self.bottleneck(p2)
        d1 = self.upconv1(b)
        d1 = torch.cat((d1, x2), dim=1)
        d1 = self.d1(d1)
        d2 = self.upconv2(d1)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.d2(d2)
        return self.sigmoid(d2)

# ==========================================
# 3. SETUP
FILE_PATH = 'data/hard/track2_data.parquet' 
MODEL_PATH = 'scripts/compass_model.pth'

TARGET_FPS = 30
PACKET_SIZE = int(500 / TARGET_FPS) 
START_SEC = 20
DURATION = 10 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initializing...")
model = MedicalUNet().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
except:
    print("⚠️ Model missing. Please train 'MedicalUNet' first.")
    exit()
model.eval()

print("Loading Data...")
df = pd.read_parquet(FILE_PATH)
if 'time_s' in df.columns: df = df.drop(columns=['time_s'])
full_data = df.values

processor = LiveProcessor(fs=500, n_channels=1024)

start_idx = START_SEC * 500
end_idx = start_idx + (DURATION * 500)
total_packets = (end_idx - start_idx) // PACKET_SIZE

print(f"Streaming {total_packets} packets (Multi-Target Mode)...")
print("Press 'q' to quit")

for i in range(total_packets):
    
    # 1. Fetch Packet
    packet_start = start_idx + (i * PACKET_SIZE)
    packet_end = packet_start + PACKET_SIZE
    raw_packet = full_data[packet_start:packet_end]
    if len(raw_packet) == 0: break

    # 2. DSP
    dsp_grid = processor.process_packet(raw_packet)
        
    # 3. AI Inference
    p99 = np.percentile(dsp_grid, 99.5)
    input_norm = np.clip(dsp_grid, 0, p99) / (p99 + 1e-9)
    
    tensor = torch.FloatTensor(input_norm).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(tensor)
        output_grid = output_tensor.squeeze().cpu().numpy()
        
    # 4. Visualization
    right_img = (output_grid * 255).astype(np.uint8)
    right_img = cv2.resize(right_img, (400, 400), interpolation=cv2.INTER_CUBIC)
    right_color = cv2.applyColorMap(right_img, cv2.COLORMAP_INFERNO)
    
    # --- MULTI-TARGET DETECTION ---
    
    # A. Threshold to find active zones (>20% confidence)
    _, mask = cv2.threshold(output_grid, 0.2, 1.0, cv2.THRESH_BINARY)
    mask_uint8 = (mask * 255).astype(np.uint8)

    # B. Find Contours (Islands)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # C. Loop through every island found
    for cnt in contours:
        # Calculate Center of Mass (Centroid) using Moments
        M = cv2.moments(cnt)
        if M["m00"] > 0: # Check area > 0 to avoid division by zero
            grid_x = M["m10"] / M["m00"]
            grid_y = M["m01"] / M["m00"]
            
            # Scale to 400x400
            screen_x = int(grid_x * 12.5 + 6)
            screen_y = int(grid_y * 12.5 + 6)
            
            # Draw Cross
            cv2.drawMarker(right_color, (screen_x, screen_y), (0, 255, 0), 
                           markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)

    # --- STATS FOOTER ---
    footer = np.zeros((20, 400, 3), dtype=np.uint8)
    combined = np.vstack((right_color, footer))
    
    current_time = START_SEC + (i / TARGET_FPS)
    stats_text = f"T={current_time:.2f}s | Targets: {len(contours)}"
    cv2.putText(combined, stats_text, (10, 412), cv2.FONT_HERSHEY_PLAIN, 1.2, (220, 220, 220), 1)
    
    # Display window
    cv2.imshow('AI-Denoised Neural Activity', combined)
    
    # Wait and check for quit
    if cv2.waitKey(int(1000/TARGET_FPS)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("✅ Done!")
