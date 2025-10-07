import cv2
import numpy as np
import matplotlib.pyplot as plt


def question3_process_image(image_path):
    print("QUESTION 3 SOLUTION")
    print("Reading RGB image")
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image_rgb.shape}")

    print("Converting to grayscale")
    grayscale_Q3 = 0.299 * image_rgb[:, :, 0] + 0.587 * \
        image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]
    grayscale_Q3 = grayscale_Q3.astype(np.uint8)

    print("Adding Gaussian noise")

    def gaussian_noise(image, mean=0, variance=20):
        sigma = np.sqrt(variance)
        noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image.astype(np.uint8), noise)
        return noisy_image

    noisy_gray = gaussian_noise(grayscale_Q3)

    print("convolution and filters")

    def Convolution_Q3(image, kernel):
        ker_h, ker_w = kernel.shape
        pad_h, pad_w = ker_h//2, ker_w//2
        padded = np.pad(
            image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = np.sum(padded[i:i+ker_h, j:j+ker_w] * kernel)
        return result

    def get_sobel_kernels(size):
        if size == 3:
            sobel_x = np.array(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        elif size == 5:
            sobel_x = np.array([[-2, -1, 0, 1, 2],
                               [-2, -1, 0, 1, 2],
                               [-4, -2, 0, 2, 4],
                               [-2, -1, 0, 1, 2],
                               [-2, -1, 0, 1, 2]], dtype=np.float32) / 16
            sobel_y = sobel_x.T
        elif size == 7:
            sobel_x = np.array([[-3, -2, -1, 0, 1, 2, 3],
                               [-3, -2, -1, 0, 1, 2, 3],
                               [-3, -2, -1, 0, 1, 2, 3],
                               [-6, -4, -2, 0, 2, 4, 6],
                               [-3, -2, -1, 0, 1, 2, 3],
                               [-3, -2, -1, 0, 1, 2, 3],
                               [-3, -2, -1, 0, 1, 2, 3]], dtype=np.float32) / 64
            sobel_y = sobel_x.T
        return sobel_x, sobel_y

    def get_laplacian_kernel(size):
        if size == 3:
            return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        elif size == 5:
            return np.array([[0, 0, -1, 0, 0],
                            [0, -1, -2, -1, 0],
                            [-1, -2, 16, -2, -1],
                            [0, -1, -2, -1, 0],
                            [0, 0, -1, 0, 0]], dtype=np.float32) / 8
        elif size == 7:
            return np.array([[0, 0, 0, -1, 0, 0, 0],
                            [0, 0, -1, -2, -1, 0, 0],
                            [0, -1, -2, -4, -2, -1, 0],
                            [-1, -2, -4, 32, -4, -2, -1],
                            [0, -1, -2, -4, -2, -1, 0],
                            [0, 0, -1, -2, -1, 0, 0],
                            [0, 0, 0, -1, 0, 0, 0]], dtype=np.float32) / 16

    print("Applying edge detection")

    sizes = [3, 5, 7]
    images = {'Clean': grayscale_Q3, 'Noisy': noisy_gray}
    results = {}

    for img_name, img in images.items():
        results[img_name] = {}
        print(f"Processing {img_name} image")

        for size in sizes:
            print(f"  Using {size}x{size} filters")

            sobel_x, sobel_y = get_sobel_kernels(size)
            laplacian = get_laplacian_kernel(size)

            sobel_x_result = Convolution_Q3(img, sobel_x)
            sobel_y_result = Convolution_Q3(img, sobel_y)

            sobel_magnitude = np.sqrt(sobel_x_result**2 + sobel_y_result**2)

            laplacian_result = Convolution_Q3(img, laplacian)

            results[img_name][size] = {
                'ORIGINAL': img,
                'sobel_x': sobel_x_result,
                'sobel_y': sobel_y_result,
                'sobel_magnitude': sobel_magnitude,
                'laplacian': laplacian_result
            }

    print("Displaying result")

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(grayscale_Q3, cmap='gray')
    plt.title('Initial Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(noisy_gray, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("results/question3_clean_&_noisy.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.suptitle('Edge Detection Results Comparison', fontsize=14)

    plot_idx = 1
    for img_name in ['Clean', 'Noisy']:
        for size in sizes:
            plt.subplot(2, 3, plot_idx)
            result = results[img_name][size]['sobel_magnitude']
            plt.imshow(result, cmap='hot')
            plt.title(f'{img_name} - {size}x{size}')
            plt.axis('off')
            plot_idx += 1
    plt.tight_layout()
    plt.savefig("results/question3_edge_detection.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    print("\nObservation Results:")
    print("-" * 30)

    for img_name in ['Clean', 'Noisy']:
        print(f"{img_name} Image:")
        for size in sizes:
            sobel_mean = np.mean(results[img_name][size]['sobel_magnitude'])
            print(f"  {size}x{size}: {sobel_mean:.1f}")

    print("\nConclusion:")
    print("- Larger filters reduce noise but it blur edges")
    print("- Smaller filters keeps details but are noise-affective")
    print("- 5x5 keeps a good balance")

    return results


if __name__ == "__main__":
    question3_process_image("my image.jpg")
