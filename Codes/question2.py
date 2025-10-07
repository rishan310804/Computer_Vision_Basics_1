import cv2
import numpy as np
import matplotlib.pyplot as plt


def question2_process_image(image_path):
    print("QUESTION 2 SOLUTION")
    print("Reading RGB image")
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image_rgb.shape}")

    print("Adding Gaussian noise")

    def gaussian_noise(image, mean=0, variance=20):
        sigma = np.sqrt(variance)
        noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image.astype(np.uint8), noise)
        return noisy_image

    noisy_rgb = gaussian_noise(image_rgb, mean=0, variance=50)

    print("Converting to grayscale")
    grayscale = 0.299 * noisy_rgb[:, :, 0] + 0.587 * \
        noisy_rgb[:, :, 1] + 0.114 * noisy_rgb[:, :, 2]
    grayscale = grayscale.astype(np.uint8)

    print("Getting the individual colour channels (R, G, B)")
    red_channel = noisy_rgb[:, :, 0]
    green_channel = noisy_rgb[:, :, 1]
    blue_channel = noisy_rgb[:, :, 2]

    print("Creating channel visualizations")
    red_image = np.zeros_like(noisy_rgb)
    red_image[:, :, 0] = red_channel

    green_image = np.zeros_like(noisy_rgb)
    green_image[:, :, 1] = green_channel

    blue_image = np.zeros_like(noisy_rgb)
    blue_image[:, :, 2] = blue_channel

    print("Creating Histogram")

    def Making_histogram(channel, bins=256):
        histogram = np.zeros(bins)
        flat_channel = channel.flatten()
        for pixel_value in flat_channel:
            if 0 <= pixel_value < bins:
                histogram[pixel_value] += 1
        return histogram

    hist_red = Making_histogram(red_channel)
    hist_green = Making_histogram(green_channel)
    hist_blue = Making_histogram(blue_channel)
    hist_gray = Making_histogram(grayscale)

    print("Showing results")
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original RGB')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(noisy_rgb)
    plt.title('Noisy RGB')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(grayscale, cmap='gray')
    plt.title('Noisy Grayscale')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(red_channel, cmap='Reds')
    plt.title('Red Channel')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(green_channel, cmap='Greens')
    plt.title('Green Channel')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("results/question2_images.png", dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(hist_red, color="red", label="Red Noisy Channel")
    plt.plot(hist_green, color="green", label="Green Noisy Channel")
    plt.plot(hist_blue, color="blue", label="Blue Noisy Channel")
    plt.plot(hist_gray, color="black", label="Grayscale Noisy")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    plt.title("Histograms (R, G, B, Grayscale) for Noisy Image")
    plt.savefig('results/question2_histograms.jpg',
                dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    question2_process_image("my image.jpg")
