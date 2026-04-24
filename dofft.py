import numpy as np
import cv2

def find_grid_offset_fft(image_path, tile_w, tile_h):
    # 1. Load image and convert to float for FFT
    img = cv2.imread(image_path, 0).astype(np.float32)
    
    # 2. Compute 2D FFT
    fft = np.fft.fft2(img)
    
    # 3. Calculate frequency coordinates for W and H
    # We look for the peak at the frequency corresponding to our period
    h, w = img.shape
    
    # Grid frequencies
    freq_x = int(w / tile_w)
    freq_y = int(h / tile_h)
    
    # Extract the complex value at the grid frequency
    # We use freq_x, freq_y to find the grid signal
    val_x = fft[0, freq_x]
    val_y = fft[freq_y, 0]
    
    # 4. Calculate phase shift
    # The phase of the complex number tells us the shift
    phase_x = np.angle(val_x)
    phase_y = np.angle(val_y)
    
    # Convert phase (-pi to pi) to pixel offset (0 to W)
    # The formula is: offset = -(phase * period) / (2 * pi)
    dx = - (phase_x * tile_w) / (2 * np.pi)
    dy = - (phase_y * tile_h) / (2 * np.pi)
    
    # Normalize result to be positive
    dx = dx % tile_w
    dy = dy % tile_h
    
    return dx, dy

# Usage
dx, dy = find_grid_offset_fft('continuous_image.png', 256, 256)
print(f"Grid Phase Shift: x={dx:.2f}, y={dy:.2f}")
