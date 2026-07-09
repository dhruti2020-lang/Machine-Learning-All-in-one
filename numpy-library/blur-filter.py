import cv2
import numpy as np

def manual_box_blur(image_path, output_path="numpy_blurred.png"):
    # 1. Load the image using OpenCV (loads as a 3D NumPy array: H x W x C)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert image data type to float32 so we don't overflow when adding pixel values
    img = img.astype(np.float32)
    height, width, channels = img.shape
    
    # Create an empty canvas of the same size to store our blurred output
    blurred_img = np.zeros_like(img)
    
    print("Slicing and blurring layers... please wait...")
    
    # 2. Slide a 3x3 window across the image using NumPy slicing
    # We start at row 1 and column 1 (instead of 0) to avoid crashing into the outer borders
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            
            # Extract a 3x3 neighborhood grid for all 3 color channels simultaneously
            # This cuts a small 3x3x3 block out of our giant 3D matrix
            neighborhood = img[y-1:y+2, x-1:x+2, :]
            
            # Calculate the mathematical average of those 9 pixels along the spatial axes (0 and 1)
            # np.mean(..., axis=(0,1)) averages the rows and columns but keeps the color channels separate!
            pixel_average = np.mean(neighborhood, axis=(0, 1))
            
            # Assign that average color recipe to our new canvas
            blurred_img[y, x, :] = pixel_average
            
    # 3. Clean up borders (copy the original sharp borders to our blurred canvas)
    blurred_img[0, :, :] = img[0, :, :]
    blurred_img[-1, :, :] = img[-1, :, :]
    blurred_img[:, 0, :] = img[:, 0, :]
    blurred_img[:, -1, :] = img[:, -1, :]
    
    # Convert data type back to standard 8-bit integers (0-255) for image saving
    blurred_img = np.clip(blurred_img, 0, 255).astype(np.uint8)
    
    # Save the output image
    cv2.imwrite(output_path, blurred_img)
    print(f"Success! Blurred image saved to: '{output_path}'")

# Run the execution
if __name__ == "__main__":
    manual_box_blur("input_image.png")