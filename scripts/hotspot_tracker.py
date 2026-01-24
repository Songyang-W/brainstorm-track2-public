"""
Hotspot Detection and Tracking Algorithm for BrainStorm Track 2

Real-time pipeline that processes neural data and computes velocity predictions
simultaneously with visualization. Uses live_video.py's filtering approach.

Pipeline:
    Raw Data (500 Hz, 1024 channels)
        │
        ▼
    LiveProcessor (Bandpass 70-150Hz + EMA smoothing)
        │
        ▼
    UNet AI Denoising
        │
        ▼
    Hotspot Detection + Velocity Prediction (computed same frame)
        │
        ▼
    Real-time OpenCV Visualization

Usage:
    python scripts/hotspot_tracker.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
from scipy.signal import butter, sosfilt
from pathlib import Path


# =============================================================================
# 1. LIVE SIGNAL PROCESSOR (from live_video.py)
# =============================================================================
class LiveProcessor:
    """Real-time signal processor with stateful filtering."""
    
    def __init__(self, fs=500, n_channels=1024):
        self.fs = fs
        self.sos = butter(4, [70, 150], btype='band', fs=fs, output='sos')
        n_sections = self.sos.shape[0]
        self.zi = np.zeros((n_sections, 2, n_channels))
        self.prev_smooth = None
        self.alpha = 0.2 

    def process_packet(self, raw_packet):
        """Process a packet of raw data and return smoothed 32x32 grid."""
        filtered, self.zi = sosfilt(self.sos, raw_packet, axis=0, zi=self.zi)
        power = filtered ** 2
        avg_power = np.mean(power, axis=0)
        grid = avg_power.reshape(32, 32)
        
        if self.prev_smooth is None:
            self.prev_smooth = grid
        else:
            self.prev_smooth = self.alpha * grid + (1 - self.alpha) * self.prev_smooth
            
        return self.prev_smooth


# =============================================================================
# 2. UNET DENOISING MODEL (from live_video.py)
# =============================================================================
class MedicalUNet(nn.Module):
    """U-Net autoencoder for denoising neural activity grids."""
    
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


# =============================================================================
# 3. VELOCITY TRACKER - Computes velocity from denoised grid
# =============================================================================
class VelocityTracker:
    """Computes cursor velocity from denoised neural activity grid."""
    
    def __init__(self, smoothing_alpha=0.3):
        self.smoothing_alpha = smoothing_alpha
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.cursor_x = 0.0
        self.cursor_y = 0.0
        self.history_x = [0.0]
        self.history_y = [0.0]
        
    def compute_velocity(self, denoised_grid):
        """
        Compute velocity from intensity-weighted center of mass.
        
        Args:
            denoised_grid: 32x32 numpy array (0-1 confidence values)
            
        Returns:
            (vx, vy, com_row, com_col, num_targets)
        """
        # Threshold to find active zones
        _, mask = cv2.threshold(denoised_grid.astype(np.float32), 0.2, 1.0, cv2.THRESH_BINARY)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours (targets)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_targets = len(contours)
        
        # Use SQUARED intensity for center of mass (exaggerates bright hotspots)
        # This makes the COM pull more strongly toward high-intensity regions
        weighted_grid = denoised_grid ** 2
        
        total_intensity = weighted_grid.sum()
        
        if total_intensity > 0.01:
            rows, cols = np.mgrid[0:32, 0:32]
            avg_row = (rows * weighted_grid).sum() / total_intensity
            avg_col = (cols * weighted_grid).sum() / total_intensity
            
            # Convert to velocity (center is 15.5, 15.5)
            # Exaggerate the displacement from center with power function
            raw_vx = (avg_col - 15.5) / 15.5
            raw_vy = -(avg_row - 15.5) / 15.5  # Negative because row 0 is top
            
            # Exaggerate small displacements (sign-preserving power < 1)
            exaggeration = 0.6  # Lower = more exaggeration
            raw_vx = np.sign(raw_vx) * (np.abs(raw_vx) ** exaggeration)
            raw_vy = np.sign(raw_vy) * (np.abs(raw_vy) ** exaggeration)
            
            # Smooth velocity
            vx = self.smoothing_alpha * raw_vx + (1 - self.smoothing_alpha) * self.prev_vx
            vy = self.smoothing_alpha * raw_vy + (1 - self.smoothing_alpha) * self.prev_vy
            
            self.prev_vx = vx
            self.prev_vy = vy
            
            return vx, vy, avg_row, avg_col, num_targets
        else:
            # Decay velocity when no activity
            self.prev_vx *= 0.9
            self.prev_vy *= 0.9
            return self.prev_vx, self.prev_vy, 15.5, 15.5, 0
    
    def update_cursor(self, vx, vy, speed_scale=5.0):
        """Update cursor position based on velocity."""
        self.cursor_x = np.clip(self.cursor_x + vx * speed_scale, -100, 100)
        self.cursor_y = np.clip(self.cursor_y + vy * speed_scale, -100, 100)
        
        self.history_x.append(self.cursor_x)
        self.history_y.append(self.cursor_y)
        
        # Keep last 100 points
        if len(self.history_x) > 100:
            self.history_x = self.history_x[-100:]
            self.history_y = self.history_y[-100:]
        
        return self.cursor_x, self.cursor_y


# =============================================================================
# 4. MAIN REAL-TIME LOOP
# =============================================================================
def load_data(difficulty="hard"):
    """Load neural data and ground truth."""
    data_path = Path(f"data/{difficulty}")
    
    neural_data = pd.read_parquet(data_path / "track2_data.parquet")
    ground_truth = pd.read_parquet(data_path / "ground_truth.parquet")
    
    # Drop time column if present
    if 'time_s' in neural_data.columns:
        neural_data = neural_data.drop(columns=['time_s'])
    
    print(f"Loaded {difficulty} dataset:")
    print(f"  Neural data: {neural_data.shape[0]} samples, {neural_data.shape[1]} channels")
    print(f"  Duration: {neural_data.shape[0] / 500:.1f} seconds")
    
    return neural_data.values, ground_truth


def run_realtime_tracking(data, ground_truth, model_path="scripts/compass_model.pth",
                          start_sec=0, duration=30, target_fps=30):
    """
    Run real-time hotspot tracking with simultaneous velocity computation.
    
    All processing happens in one loop - no separate filtering step.
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    packet_size = int(500 / target_fps)  # ~16 samples per frame at 30fps
    
    # Pre-smooth ground truth (it's at 500Hz, we display at 30fps)
    # Average over each packet window for fair comparison
    gt_vx_smooth = np.convolve(ground_truth['vx'].values, np.ones(packet_size)/packet_size, mode='same')
    gt_vy_smooth = np.convolve(ground_truth['vy'].values, np.ones(packet_size)/packet_size, mode='same')
    
    # Load model
    print(f"\nLoading UNet model from {model_path}...")
    model = MedicalUNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"  ✓ Model loaded (device: {device})")
    except Exception as e:
        print(f"  ⚠️ Model load failed: {e}")
        return
    
    # Initialize processors
    dsp_processor = LiveProcessor(fs=500, n_channels=1024)
    velocity_tracker = VelocityTracker(smoothing_alpha=0.3)
    
    # Calculate indices
    start_idx = start_sec * 500
    end_idx = start_idx + (duration * 500)
    total_packets = (end_idx - start_idx) // packet_size
    
    print(f"\nStarting real-time tracking:")
    print(f"  Time range: {start_sec}s to {start_sec + duration}s")
    print(f"  Total packets: {total_packets}")
    print(f"  Press 'q' to quit\n")
    
    # Storage for analysis
    predictions = {'time': [], 'vx': [], 'vy': [], 'gt_vx': [], 'gt_vy': [], 'cursor_x': [], 'cursor_y': []}
    
    for i in range(total_packets):
        # ==========================================
        # STEP 1: Fetch raw packet
        # ==========================================
        packet_start = start_idx + (i * packet_size)
        packet_end = packet_start + packet_size
        raw_packet = data[packet_start:packet_end]
        if len(raw_packet) == 0:
            break
        
        current_time = start_sec + (i / target_fps)
        
        # ==========================================
        # STEP 2: DSP filtering (bandpass + EMA)
        # ==========================================
        dsp_grid = dsp_processor.process_packet(raw_packet)
        
        # ==========================================
        # STEP 3: AI denoising (UNet inference)
        # ==========================================
        p99 = np.percentile(dsp_grid, 99.5)
        input_norm = np.clip(dsp_grid, 0, p99) / (p99 + 1e-9)
        
        tensor = torch.FloatTensor(input_norm).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(tensor)
            denoised_grid = output_tensor.squeeze().cpu().numpy()
        
        # ==========================================
        # STEP 4: Velocity computation (SIMULTANEOUS)
        # ==========================================
        vx, vy, com_row, com_col, num_targets = velocity_tracker.compute_velocity(denoised_grid)
        cursor_x, cursor_y = velocity_tracker.update_cursor(vx, vy)
        
        # Get smoothed ground truth for this packet
        gt_idx = packet_start + packet_size // 2
        gt_vx_val = gt_vx_smooth[gt_idx] if gt_idx < len(gt_vx_smooth) else 0
        gt_vy_val = gt_vy_smooth[gt_idx] if gt_idx < len(gt_vy_smooth) else 0
        
        # Store predictions and ground truth
        predictions['time'].append(current_time)
        predictions['vx'].append(vx)
        predictions['vy'].append(vy)
        predictions['gt_vx'].append(gt_vx_val)
        predictions['gt_vy'].append(gt_vy_val)
        predictions['cursor_x'].append(cursor_x)
        predictions['cursor_y'].append(cursor_y)
        
        # ==========================================
        # STEP 5: Visualization (real-time OpenCV)
        # ==========================================
        
        PANEL_SIZE = 300
        HEADER_H = 25
        scale = PANEL_SIZE / 32.0
        
        # --- PANEL 1: RAW DSP (Unfiltered - before UNet) ---
        # Normalize raw DSP grid for display
        raw_display = input_norm  # This is the normalized DSP output before UNet
        raw_img = (raw_display * 255).astype(np.uint8)
        raw_img = cv2.resize(raw_img, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_CUBIC)
        raw_color = cv2.applyColorMap(raw_img, cv2.COLORMAP_VIRIDIS)
        
        # Compute center of mass for RAW data
        raw_total = input_norm.sum()
        if raw_total > 0.01:
            rows, cols = np.mgrid[0:32, 0:32]
            raw_com_row = (rows * input_norm).sum() / raw_total
            raw_com_col = (cols * input_norm).sum() / raw_total
        else:
            raw_com_row, raw_com_col = 15.5, 15.5
        
        # Draw raw center of mass (red cross)
        raw_screen_x = int((raw_com_col + 0.5) * scale)
        raw_screen_y = int((raw_com_row + 0.5) * scale)
        cv2.drawMarker(raw_color, (raw_screen_x, raw_screen_y), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        
        header_raw = np.zeros((HEADER_H, PANEL_SIZE, 3), dtype=np.uint8)
        header_raw[:] = (40, 40, 40)
        cv2.putText(header_raw, "RAW DSP (Before UNet)", (5, 17), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        panel_raw = np.vstack((header_raw, raw_color))
        
        # --- PANEL 2: AI DENOISED (Filtered - after UNet) ---
        neural_img = (denoised_grid * 255).astype(np.uint8)
        neural_img = cv2.resize(neural_img, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_CUBIC)
        neural_color = cv2.applyColorMap(neural_img, cv2.COLORMAP_INFERNO)
        
        # Draw center of mass marker (yellow cross) - uses denoised data
        screen_x = int((com_col + 0.5) * scale)
        screen_y = int((com_row + 0.5) * scale)
        cv2.drawMarker(neural_color, (screen_x, screen_y), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        
        # Draw detected targets (green circles)
        _, mask = cv2.threshold(denoised_grid.astype(np.float32), 0.5, 1.0, cv2.THRESH_BINARY)
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                grid_x = M["m10"] / M["m00"]
                grid_y = M["m01"] / M["m00"]
                gx = int((grid_x + 0.5) * scale)
                gy = int((grid_y + 0.5) * scale)
                cv2.circle(neural_color, (gx, gy), 5, (0, 255, 0), 2)
        
        header_ai = np.zeros((HEADER_H, PANEL_SIZE, 3), dtype=np.uint8)
        header_ai[:] = (40, 40, 40)
        cv2.putText(header_ai, "AI DENOISED (After UNet)", (5, 17), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        panel_ai = np.vstack((header_ai, neural_color))
        
        # --- PANEL 3: Cursor Trajectory ---
        cursor_img = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
        cursor_img[:] = (25, 25, 25)
        
        center = PANEL_SIZE // 2
        cv2.line(cursor_img, (center, 0), (center, PANEL_SIZE), (50, 50, 50), 1)
        cv2.line(cursor_img, (0, center), (PANEL_SIZE, center), (50, 50, 50), 1)
        cv2.rectangle(cursor_img, (20, 20), (PANEL_SIZE-20, PANEL_SIZE-20), (60, 60, 60), 1)
        
        cursor_scale = (PANEL_SIZE - 40) / 200
        history_x = velocity_tracker.history_x
        history_y = velocity_tracker.history_y
        for j in range(1, len(history_x)):
            pt1_x = int(history_x[j-1] * cursor_scale + center)
            pt1_y = int(-history_y[j-1] * cursor_scale + center)
            pt2_x = int(history_x[j] * cursor_scale + center)
            pt2_y = int(-history_y[j] * cursor_scale + center)
            alpha = j / len(history_x)
            color = (int(80 * alpha), int(150 * alpha), int(255 * alpha))
            cv2.line(cursor_img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, 2)
        
        cursor_screen_x = int(cursor_x * cursor_scale + center)
        cursor_screen_y = int(-cursor_y * cursor_scale + center)
        cv2.circle(cursor_img, (cursor_screen_x, cursor_screen_y), 10, (0, 180, 255), -1)
        cv2.circle(cursor_img, (cursor_screen_x, cursor_screen_y), 10, (255, 255, 255), 2)
        
        header_cursor = np.zeros((HEADER_H, PANEL_SIZE, 3), dtype=np.uint8)
        header_cursor[:] = (40, 40, 40)
        cv2.putText(header_cursor, f"CURSOR ({cursor_x:.0f}, {cursor_y:.0f})", (5, 17), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        panel_cursor = np.vstack((header_cursor, cursor_img))
        
        # --- PANEL 4: Velocity Arrow ---
        vel_img = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
        vel_img[:] = (25, 25, 25)
        
        cv2.line(vel_img, (center, 0), (center, PANEL_SIZE), (50, 50, 50), 1)
        cv2.line(vel_img, (0, center), (PANEL_SIZE, center), (50, 50, 50), 1)
        cv2.circle(vel_img, (center, center), int(center * 0.8), (40, 40, 40), 1)
        
        arrow_scale = center * 0.8
        arrow_end_x = int(center + vx * arrow_scale)
        arrow_end_y = int(center - vy * arrow_scale)
        cv2.arrowedLine(vel_img, (center, center), (arrow_end_x, arrow_end_y), 
                        (255, 255, 0), 3, tipLength=0.2)
        
        header_vel = np.zeros((HEADER_H, PANEL_SIZE, 3), dtype=np.uint8)
        header_vel[:] = (40, 40, 40)
        cv2.putText(header_vel, f"VELOCITY ({vx:.2f}, {vy:.2f})", (5, 17), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
        panel_vel = np.vstack((header_vel, vel_img))
        
        # --- Combine: Top row (Raw | AI), Bottom row (Cursor | Velocity) ---
        top_row = np.hstack((panel_raw, panel_ai))
        bottom_row = np.hstack((panel_cursor, panel_vel))
        combined = np.vstack((top_row, bottom_row))
        
        # --- Stats footer ---
        footer = np.zeros((25, combined.shape[1], 3), dtype=np.uint8)
        footer[:] = (30, 30, 30)
        stats = f"HARD Dataset | T={current_time:.2f}s | Targets={num_targets} | FPS={target_fps}"
        cv2.putText(footer, stats, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        combined = np.vstack((combined, footer))
        
        # Display
        cv2.imshow('Real-Time Hotspot Tracking', combined)
        
        if cv2.waitKey(int(1000/target_fps)) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Return predictions for analysis
    return pd.DataFrame(predictions)


def analyze_predictions(predictions, ground_truth):
    """Analyze prediction accuracy and plot comparison."""
    from scipy.stats import pearsonr
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 50)
    print("ANALYSIS")
    print("=" * 50)
    
    # Use the smoothed ground truth that was stored during tracking
    gt_vx = np.array(predictions['gt_vx'])
    gt_vy = np.array(predictions['gt_vy'])
    
    # Normalize ground truth to same scale (GT ranges ~-300 to +300)
    gt_vx_norm = gt_vx / 300
    gt_vy_norm = gt_vy / 300
    
    # Calculate correlations
    corr_vx, _ = pearsonr(predictions['vx'], gt_vx_norm)
    corr_vy, _ = pearsonr(predictions['vy'], gt_vy_norm)
    
    print(f"Velocity Correlation:")
    print(f"  Vx: {corr_vx:.3f}")
    print(f"  Vy: {corr_vy:.3f}")
    print(f"  Combined: {(corr_vx + corr_vy) / 2:.3f}")
    
    # --- Plot comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Predicted vs Ground Truth Velocity (Smoothed)', fontsize=14, fontweight='bold')
    
    time = predictions['time']
    
    # Vx plot
    axes[0].plot(time, gt_vx_norm, 'r-', alpha=0.7, linewidth=1.5, label=f'Ground Truth Vx')
    axes[0].plot(time, predictions['vx'], 'b-', alpha=0.7, linewidth=1.5, label=f'Predicted Vx')
    axes[0].set_ylabel('Vx (normalized)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'X Velocity - Correlation: {corr_vx:.3f}')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Vy plot
    axes[1].plot(time, gt_vy_norm, 'r-', alpha=0.7, linewidth=1.5, label=f'Ground Truth Vy')
    axes[1].plot(time, predictions['vy'], 'b-', alpha=0.7, linewidth=1.5, label=f'Predicted Vy')
    axes[1].set_ylabel('Vy (normalized)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'Y Velocity - Correlation: {corr_vy:.3f}')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return corr_vx, corr_vy


if __name__ == "__main__":
    print("=" * 60)
    print("BrainStorm Track 2 - Real-Time Hotspot Tracking")
    print("=" * 60)
    print("\nPipeline: Raw → DSP Filter → UNet → Velocity (all in one loop)")
    
    # Load data
    data, ground_truth = load_data("hard")
    
    # Run real-time tracking
    predictions = run_realtime_tracking(
        data, 
        ground_truth,
        model_path="scripts/compass_model.pth",
        start_sec=0,
        duration=30,
        target_fps=30
    )
    
    if predictions is not None:
        # Analyze results
        analyze_predictions(predictions, ground_truth)
    
    print("\n✅ Done!")
