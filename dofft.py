import numpy as np
import cv2

def find_offset_fft2(image_path, tile_w, tile_h):
    # 1. Load and prepare image (Grayscale)
    img = cv2.imread(image_path, 0).astype(np.float32)
    h, w = img.shape

    # 2. Perform FFT
    # fft2 returns complex numbers representing frequency and phase
    f_transform = np.fft.fft2(img)

    # 3. Identify the Grid Frequency Bin
    # The grid pattern creates a signal peak at the frequency:
    # freq_x = image_width / tile_width
    # freq_y = image_height / tile_height
    # We use integer division to target the specific bin
    bin_x = int(w / tile_w)
    bin_y = int(h / tile_h)

    # 4. Extract Phase at these bins
    # We look at the complex value at [0, bin_x] for horizontal and [bin_y, 0] for vertical
    # These represent the primary frequency component of the repeating tiles
    val_x = f_transform[0, bin_x]
    val_y = f_transform[bin_y, 0]

    # np.angle returns values in radians (-pi to pi)
    phase_x = np.angle(val_x)
    phase_y = np.angle(val_y)

    # 5. Convert Phase to Pixel Offset
    # Formula: offset = -(phase * period) / (2 * pi)
    # We add the period before modulo to ensure result is positive
    dx = (-phase_x * tile_w) / (2 * np.pi)
    dy = (-phase_y * tile_h) / (2 * np.pi)

    return dx % tile_w, dy % tile_h

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

def find_grid_offset(image_path):
    # 1. Load and prepare the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Canny to highlight edges
    edges = cv2.Canny(gray, 50, 150)
    
    # 2. Calculate Projections
    # Sum pixels along columns (x-axis) and rows (y-axis)
    x_projection = np.sum(edges, axis=0)
    y_projection = np.sum(edges, axis=1)
    
    # 3. Find the first significant peak
    # We define a threshold to ignore noise (e.g., 20% of the max peak)
    x_threshold = np.max(x_projection) * 0.2
    y_threshold = np.max(y_projection) * 0.2
    
    # Find indices where projection exceeds threshold
    x_indices = np.where(x_projection > x_threshold)[0]
    y_indices = np.where(y_projection > y_threshold)[0]
    
    # The first value is your offset
    offset_x = x_indices[0] if len(x_indices) > 0 else 0
    offset_y = y_indices[0] if len(y_indices) > 0 else 0
    
    return offset_x, offset_y

def find_terraria_grid_offset(image_path, tile_size=16):
    # Load as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Edge detection: This highlights the boundaries of the dirt/grass blocks
    # We use Laplacian for a generic edge response
    edges = cv2.Laplacian(img, cv2.CV_64F)
    edges = np.abs(edges)
    
    best_score = -1
    best_offset = (0, 0)
    
    h, w = edges.shape
    
    # 2. Iterate through all 256 possible offsets (0-15)
    for dy in range(tile_size):
        for dx in range(tile_size):
            # Calculate the score for this offset
            # We sum the edge intensity along the lines where grid boundaries exist
            # Vertical lines: x = dx, dx + 16, dx + 32...
            # Horizontal lines: y = dy, dy + 16, dy + 32...
            
            # Using slicing for speed instead of loops
            vertical_score = np.sum(edges[:, dx::tile_size])
            horizontal_score = np.sum(edges[dy::tile_size, :])
            
            total_score = vertical_score + horizontal_score
            
            if total_score > best_score:
                best_score = total_score
                best_offset = (dx, dy)
                
    return best_offset
def find_terraria_grid_optimized(image_path, tile_size=16):
    # 1. Load image and crop to a 'safe' area if possible
    # (Avoid the UI/Inventory bar at the top)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0, 0
    img = img[100:-10, 10:-10] # Basic crop to remove HUD/UI
    
    # 2. Isolate high-contrast, axis-aligned edges
    # Sobel X for vertical lines, Sobel Y for horizontal lines
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Thresholding: Only count edges that are clearly visible
    # Adjust 50 based on how "noisy" your screenshot is
    _, edges_v = cv2.threshold(np.abs(sobel_x), 50, 255, cv2.THRESH_BINARY)
    _, edges_h = cv2.threshold(np.abs(sobel_y), 50, 255, cv2.THRESH_BINARY)
    
    best_score = -1
    best_offset = (0, 0)
    
    # 3. Voting: Count matches instead of summing intensities
    for dy in range(tile_size):
        for dx in range(tile_size):
            # Sum the binary '1's found exactly on the grid lines
            v_matches = np.sum(edges_v[:, dx::tile_size])
            h_matches = np.sum(edges_h[dy::tile_size, :])
            
            total_score = v_matches + h_matches
            
            if total_score > best_score:
                best_score = total_score
                best_offset = (dx, dy)
                
    return best_offset

def find_offset_by_downsampling(image_path, tile_size=16):
    # 1. Load image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. THE SECRET: Downsample and Blur
    # We shrink the image by a factor of 4. This effectively turns a 16x16
    # block into a 4x4 'pixel', washing away all the internal texture noise.
    scale = 4
    small_img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_AREA)
    blurred_img = cv2.GaussianBlur(small_img, (5, 5), 0)

    # We only care about the grid alignment, which is tile_size / scale
    scaled_tile_size = tile_size // scale

    best_score = float('inf')
    best_offset = (17, 17)

    # 3. Sliding Window Autocorrelation
    # We check offsets from 0 to the scaled tile size
    for dy in range(scaled_tile_size):
        for dx in range(scaled_tile_size):

            # Create a shifted version of the image
            # We shift by dx, dy and compare the difference
            shifted_img = np.roll(blurred_img, shift=(dy, dx), axis=(0, 1))

            # The 'error' is the difference between the image and its shifted version
            # If the offset is correct, the difference will be minimized
            error = np.sum((blurred_img.astype(float) - shifted_img.astype(float))**2)

            if error < best_score:
                best_score = error
                best_offset = (dx * scale, dy * scale) # Scale back up to original pixels

    return best_offset

def find_terraria_offset_correct(image_path, tile_size=16):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Get the gradient magnitude (shows all sharp changes/seams)
    # Using Sobel to get the strength of changes
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    
    best_score = -1
    best_offset = (0, 0)
    
    h, w = gradient.shape
    
    # 2. Iterate through all 16x16 possible offsets
    for dy in range(tile_size):
        for dx in range(tile_size):
            # We are summing the gradient intensity along the grid lines
            # Grid lines occur at x = dx, dx + 16, dx + 32...
            # and y = dy, dy + 16, dy + 32...
            
            # Sum vertical lines
            # This slices the gradient array to only look at the vertical grid columns
            v_score = np.sum(gradient[:, dx::tile_size])
            
            # Sum horizontal lines
            # This slices the gradient array to only look at the horizontal grid rows
            h_score = np.sum(gradient[dy::tile_size, :])
            
            score = v_score + h_score
            
            if score > best_score:
                best_score = score
                best_offset = (dx, dy)
                
    return best_offset

def find_offset_by_variance(image_path, tile_size=16):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # 1. Calculate Local Variance
    # This creates a "heat map" where bright areas are high-detail (boundaries)
    # and dark areas are low-detail (solid block centers).
    mean = cv2.boxFilter(img, -1, (3, 3))
    mean_sq = cv2.boxFilter(img**2, -1, (3, 3))
    variance = mean_sq - mean**2

    # 2. Iterate offsets
    best_score = -1
    best_offset = (0, 0)

    h, w = variance.shape

    # If the user is zoomed in, tile_size needs to be 32, 48, etc.
    for dy in range(tile_size):
        for dx in range(tile_size):
            # Sum variance along grid lines
            # Grid lines at: x = dx, dx+16, ... and y = dy, dy+16, ...
            # We use slicing to grab the "grid" rows and columns
            v_sum = np.sum(variance[:, dx::tile_size])
            h_sum = np.sum(variance[dy::tile_size, :])

            score = v_sum + h_sum

            if score > best_score:
                best_score = score
                best_offset = (dx, dy)

    return best_offset

imPaths = ['./captureData/imageFrames/199.png', './captureData/imageFrames/264.png', './captureData/imageFrames/310.png']
# Usage
for imPath in imPaths:
    dx, dy = find_offset_by_variance(imPath)
    print(f"{imPath} Refined Offset: x={dx}, y={dy}")
exit(0)
# Usage
offset_x, offset_y = find_grid_offset(imPath)
print(f"Grid Offset: ({offset_x}, {offset_y})")
# Usage
dx, dy = find_grid_offset_fft(imPath, 16, 16)
print(f"Grid Phase Shift: x={dx:.2f}, y={dy:.2f}")

dx, dy = find_offset_fft2(imPath, 16, 16)
print(f"Grid Phase Shift: x={dx:.2f}, y={dy:.2f}")
print(f'Adjusted shift: x={dx-5:.2f}, y={dy+1.5:.2f}')
dx, dy = find_terraria_grid_offset(imPath, 16)
print(f"Grid Phase TShift: x={dx:.2f}, y={dy:.2f}")
