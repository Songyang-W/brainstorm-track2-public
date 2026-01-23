import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from neural_preprocessor import NeuralPreprocessor
from denoising_autoencoder import DenoisingAutoencoder

# --- Configuration ---
FILE_PATH = '../data/easy/track2_data.parquet'
BATCH_SIZE = 32
EPOCHS = 5

# 1. Load Data
print("Loading Parquet file...")
df = pd.read_parquet(FILE_PATH)

# Drop Time column if it exists, ensure we only have 1024 channels
if 'time_s' in df.columns:
    df = df.drop(columns=['time_s'])
    
raw_values = df.values # Convert to numpy

# 2. Preprocess (DSP)
print("Running DSP (Bandpass + Envelope + Reshape)...")
preprocessor = NeuralPreprocessor(fs=500)
processed_frames = preprocessor.process_chunk(raw_values)

# Convert to PyTorch Tensor
tensor_data = torch.FloatTensor(processed_frames)
dataset = TensorDataset(tensor_data, tensor_data) # Input = Target (Autoencoder)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss() # Compare Output vs Input
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train Loop
print(f"Starting Training on {device}...")
model.train()

for epoch in range(EPOCHS):
    running_loss = 0.0
    for data in loader:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Optional: Add synthetic noise to inputs to make it robust
        # noise = torch.randn_like(inputs) * 0.1
        # noisy_inputs = inputs + noise
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(loader):.6f}")

# 5. Save the trained brain
torch.save(model.state_dict(), "compass_model.pth")
print("Model Saved!")