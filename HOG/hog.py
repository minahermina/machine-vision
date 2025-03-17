"""
Required: 
    [x] 1- Pre-process the image with the appropriate aspect ratio.
    [x] 2- Get image gradients by convoluting sobel kernel (Applying 1st order derivative) with the image.
    [x] 3- Get magnitude and phase for the image.
    [x] 4- Get histogram for the magnitudes based on the directions using the ratio approach.
    [x] 5 - Normalize each histogram.
    [x] 6- Concatenate all histograms into a single feature vector.  

Bonus:
    [x] 1- Normalize each 4 histograms together in 16*16 block instead of normalizing each histogram alone in the 8*8 block. 
    [] 2- Draw the HoG over the image (Magnitude & Phase).

"""
import numpy as np
# from python import matplotlib.pyplot as plt
# from python import cv2
# from python import sys

import matplotlib.pyplot as plt
import cv2
import sys

def img_disp(image: np.ndarray):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.show()

dimensions = (64, 128)


def validate_dimensions(dimensions):
    width, height = dimensions

    if width * 2 != height:
        raise ValueError(f"Dimensions {dimensions} do not have a 1:2 ratio.")

    if width % 8 != 0 or height % 8 != 0:
        raise ValueError(f"Dimensions {dimensions} are not divisible by 8.")

    print(f"    Dimensions {dimensions} are valid.")
    return True

def preprocess_image(image):
    # Validate dimensions before resizing
    validate_dimensions(dimensions)
    print(f'    image before resize: {image.shape}')
    image = cv2.resize(image, dimensions, interpolation=cv2.INTER_CUBIC)
    print(f'    image after resize: {image.shape}')
    return image

def get_gradients(image) -> tuple[np.ndarray, np.ndarray]:
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ])

    grad_x = cv2.filter2D(image, -1, sobel_x)
    grad_y = cv2.filter2D(image, -1, sobel_y)
    print(f'    Shape gradient matrices: {grad_x.shape}')

    return grad_x, grad_y

def calc_magnitude_phase(grad_x: np.ndarray, grad_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))
    # Convert gradient phase from radians to degrees in range 0-180°
    phase = np.rad2deg(np.arctan2(grad_y, grad_x)) #% 180.0

    print(f'    Shape of magnitude matrix: {magnitude.shape}')
    print(f'    Shape of phase matrix: {phase.shape} ')

    return magnitude, phase

def calc_histogram(magnitude, phase, cell_size=8, nbins=9):
    # Get image dimensions
    height, width = magnitude.shape

    # Calculate number of cells
    n_cells_y = height // cell_size
    n_cells_x = width // cell_size

    # histogram array
    # - n_cells_y: number of cells in the vertical direction (height/cell_size)
    # - n_cells_x: number of cells in the horizontal direction (width/cell_size)
    # - nbins: number of orientation bins (typically 9 for HOG)
    histograms = np.zeros((n_cells_y, n_cells_x, nbins))

    bin_width = 180.0 / nbins

    # For each cell
    for y in range(n_cells_y):
        for x in range(n_cells_x):
            # Get cell region
            cell_mag = magnitude[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
            cell_phase = phase[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]

            # For each pixel in the cell
            for i in range(cell_size):
                for j in range(cell_size):
                    # Get magnitude and phase for this pixel
                    mag = cell_mag[i, j]
                    angle = cell_phase[i, j]

                    # 
                    bin_idx = int(angle // bin_width)
                    bin_idx_next = (bin_idx + 1) % nbins

                    # Calculate weight based on distance to bin centers
                    bin_center = bin_idx * bin_width + bin_width / 2
                    weight_next = (angle - bin_center + bin_width/2) / bin_width
                    weight_next = max(0, min(1, weight_next))  # Ensure weight is between 0 and 1
                    weight = 1 - weight_next

                    # Add weighted magnitude to histogram bins
                    histograms[y, x, bin_idx] += weight * mag
                    histograms[y, x, bin_idx_next] += weight_next * mag

    return histograms

def normalize_histograms(histograms: np.ndarray, block_size:int = 2) -> np.ndarray:
    """
    Default is 2x2 cells (16x16 pixel blocks).
    (n_cells_y - block_size + 1, n_cells_x - block_size + 1, block_size * block_size * nbins)
    """
    n_cells_y, n_cells_x, nbins = histograms.shape

    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1
    block_features = block_size * block_size * nbins

    normalized_blocks = np.zeros((n_blocks_y, n_blocks_x, block_features))

    # For each block of cells
    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            # Extract the block of histograms
            block = histograms[y:y+block_size, x:x+block_size, :]

            # Reshape block to a 1D vector
            block_vector = block.reshape(-1)

            # L2 normalization
            epsilon = 1e-8
            l2_norm = np.sqrt(np.sum(block_vector**2) + epsilon)

            # Store the normalized block vector
            if l2_norm > 0:
                normalized_blocks[y, x, :] = block_vector / l2_norm

    return normalized_blocks

def hog(image: np.ndarray) -> np.ndarray:
    # 1- Pre-process the image with the appropriate aspect ratio(64x128).
    print(f"Step 1: resizing images to {dimensions}")
    image = preprocess_image(image)
    img_disp(image)

    # 2- Get image gradients by convoluting sobel kernel (Applying 1st order derivative) with the image.
    print()
    print(f"Step 2: Get image gradients by convoluting sobel kernel")
    grad_x, grad_y = get_gradients(image)

    #3- Get magnitude and phase for the image.
    print()
    print(f"Step 3: Get magnitude and phase for the image.")
    magnitude, phase  = calc_magnitude_phase(grad_y, grad_x)

    # 4- Get histogram for the magnitudes based on the directions using the ratio approach.
    print()
    print(f"Step 4: Get histogram for the magnitudes based on the directions using the ratio approach.")
    histograms = calc_histogram(magnitude, phase, cell_size=8, nbins=9) 
    print(f'    histograms shape: {histograms.shape}')

    # 5 - Normalize each histogram.
    normalized_blocks = normalize_histograms(histograms, block_size=1) 
    print(f'normalized_blocks : {normalized_blocks .shape}')

    # 6- Concatenate all histograms into a single feature vector (with bouns 1)
    feature_vector = normalized_blocks.flatten()
    print(f'feature_vector: {feature_vector.shape}')

    return feature_vector


def main() -> None:
    args = sys.argv[:]
    # getting image path
    if len(args) < 2:
        print(f"Usage: ./{args[0]} <path-to-image-file>")
        sys.exit(0)

    image_paths = sys.argv[1:]
    images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

    for image in images:
        feature_vector = hog(image)

if __name__ == '__main__':
    main()
