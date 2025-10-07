import cv2
import numpy as np
import matplotlib.pyplot as plt


def question1_process_image(image_path):
    print("QUESTION 1 SOLUTION")
    print("Reading RGB image")
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image_rgb.shape}")

    print("Converting to grayscale")
    grayscale = 0.299 * image_rgb[:, :, 0] + 0.587 * \
        image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]
    grayscale = grayscale.astype(np.uint8)

    print("Getting the individual colour channels (R, G, B)")
    red_channel = image_rgb[:, :, 0]
    green_channel = image_rgb[:, :, 1]
    blue_channel = image_rgb[:, :, 2]

    print("Creating channel visualizations")
    red_image = np.zeros_like(image_rgb)
    red_image[:, :, 0] = red_channel

    green_image = np.zeros_like(image_rgb)
    green_image[:, :, 1] = green_channel

    blue_image = np.zeros_like(image_rgb)
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
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original RGB')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(grayscale, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(red_channel, cmap='Reds')
    plt.title('Red Channel')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(green_channel, cmap='Greens')
    plt.title('Green Channel')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("results/question1_images.png", dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(hist_red, color="red", label="Red Channel")
    plt.plot(hist_green, color="green", label="Green Channel")
    plt.plot(hist_blue, color="blue", label="Blue Channel")
    plt.plot(hist_gray, color="black", label="Grayscale")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    plt.title("Histograms (R, G, B, Grayscale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/question1_histograms.png',
                dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    question1_process_image("my image.jpg")
