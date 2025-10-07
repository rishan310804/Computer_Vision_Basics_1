import cv2
import numpy as np
import matplotlib.pyplot as plt


def question4_process_image(image_path):
    print("QUESTION 4 SOLUTION")
    print("Reading RGB image")
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image_rgb.shape}")

    print("Converting to grayscale")
    grayscale_Q4 = 0.299 * image_rgb[:, :, 0] + 0.587 * \
        image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]
    grayscale_Q4 = grayscale_Q4.astype(np.uint8)

    print("Adding Gaussian noise")

    def gaussian_noise(image, mean=0, variance=20):
        sigma = np.sqrt(variance)
        noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image.astype(np.uint8), noise)
        return noisy_image

    noisy_gray = gaussian_noise(grayscale_Q4)

    print("Setting Canny edge detection functions")

    def Convolution_Q4(image, kernel):
        ker_h, ker_w = kernel.shape
        pad_h, pad_w = ker_h//2, ker_w//2
        padded = np.pad(
            image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = np.sum(padded[i:i+ker_h, j:j+ker_w] * kernel)
        return result

    print("Gaussian Smoothing")

    def gaussian_smoothing(image, kernel_size=5, sigma=1.0):
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * \
                    np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        return Convolution_Q4(image, kernel)

    print("Computing gradient magnitude and orientation using Sobel filter")

    def compute_gradients(image):
        sobel_x = np.array(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        gradient_x = Convolution_Q4(image, sobel_x)
        gradient_y = Convolution_Q4(image, sobel_y)

        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
        direction[direction < 0] += 180

        return gradient_x, gradient_y, magnitude, direction

    print("non maximum suppression")

    def non_maximum_suppression(magnitude, direction):
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                current_magnitude = magnitude[i, j]
                current_angle = direction[i, j]

                if (0 <= current_angle < 22.5) or (157.5 <= current_angle <= 180):
                    neighbor1 = magnitude[i, j-1]
                    neighbor2 = magnitude[i, j+1]
                elif 22.5 <= current_angle < 67.5:
                    neighbor1 = magnitude[i-1, j+1]
                    neighbor2 = magnitude[i+1, j-1]
                elif 67.5 <= current_angle < 112.5:
                    neighbor1 = magnitude[i-1, j]
                    neighbor2 = magnitude[i+1, j]
                else:
                    neighbor1 = magnitude[i-1, j-1]
                    neighbor2 = magnitude[i+1, j+1]

                if current_magnitude >= neighbor1 and current_magnitude >= neighbor2:
                    suppressed[i, j] = current_magnitude
                else:
                    suppressed[i, j] = 0

        return suppressed

    print("Hysteresis thresholding")

    def hysteresis_thresholding(image, low_threshold=50, high_threshold=150):
        height, width = image.shape

        strong_edges = (image >= high_threshold)
        weak_edges = ((image >= low_threshold) & (image < high_threshold))

        edges = np.zeros_like(image, dtype=np.uint8)
        edges[strong_edges] = 255
        edges[weak_edges] = 127

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        changed = True

        while changed:
            changed = False
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    if edges[i, j] == 127:
                        for di, dj in directions:
                            if edges[i + di, j + dj] == 255:
                                edges[i, j] = 255
                                changed = True
                                break

        edges[edges == 127] = 0
        return edges

    print("Canny edge detection")

    def canny_edge_detection(image, low_threshold=50, high_threshold=150):
        smoothed = gaussian_smoothing(image, kernel_size=5, sigma=1.0)
        grad_x, grad_y, magnitude, direction = compute_gradients(smoothed)
        suppressed = non_maximum_suppression(magnitude, direction)
        final_edges = hysteresis_thresholding(
            suppressed, low_threshold, high_threshold)

        return {
            'original': image,
            'smoothed': smoothed,
            'gradient_x': grad_x,
            'gradient_y': grad_y,
            'magnitude': magnitude,
            'suppressed': suppressed,
            'final_edges': final_edges
        }

    print("Applying Canny edge detection")

    images = {'Clean': grayscale_Q4, 'Noisy': noisy_gray}
    results = {}

    for img_name, img in images.items():
        print(f"Processing {img_name} image")
        results[img_name] = canny_edge_detection(
            img, low_threshold=50, high_threshold=150)

    print("Displaying Canny edge detection results")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(results['Clean']['final_edges'], cmap='gray')
    plt.title('Clean Final Edges')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(results['Noisy']['final_edges'], cmap='gray')
    plt.title('Noisy Final Edges')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("results/question4_Canny_Edge.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    print("\nObservation:")
    clean_edges = np.sum(results['Clean']['final_edges'] > 0)
    noisy_edges = np.sum(results['Noisy']['final_edges'] > 0)
    change = ((noisy_edges - clean_edges) / clean_edges) * 100

    print(f"Clean image edges: {clean_edges}")
    print(f"Noisy image edges: {noisy_edges}")
    print(f"Change due to noise: {change:+.1f}%")

    print("\nConclusion:")
    print("- Noise will increase false edge detection")
    print("- Canny algorithm can handle noise at a reasonable way")

    return results


if __name__ == "__main__":
    question4_process_image("my image.jpg")
