import numpy as np
from scipy.ndimage import affine_transform
from tifffile import imsave
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


class SimData:
    def __init__(self, frames, X, Y, n_blobs=10, noise_amplitude=1.0, blob_amplitude=1.0,
                 max_drift=0.5, max_jitter=2.0, background_noise=1.0, shot_noise=0.1):
        """
        Initialize the SimData object with the given parameters.

        Parameters:
        - frames: Number of frames in the simulation.
        - X, Y: Dimensions of the images.
        - n_blobs: Number of gaussian blobs (peaks or valleys) to add to the base image.
        - noise_amplitude: Amplitude of the noise in the base image.
        - blob_amplitude: Amplitude of the gaussian blobs.
        - max_drift: Maximum drift value or tuple for quadratic drift.
        - max_jitter: Maximum random jitter in the image per frame.
        - background_noise: Constant background noise level.
        - shot_noise: Frame-by-frame noise level.
        """

        # Basic parameters
        self.frames = frames
        self.X = X
        self.Y = Y
        self.n_blobs = n_blobs
        self.noise_amplitude = noise_amplitude
        self.blob_amplitude = blob_amplitude
        self.data = None
        self.shifts = None
        self.max_jitter = max_jitter
        self.background_noise = background_noise
        self.shot_noise = shot_noise

        # Determine if the drift is linear or quadratic, and randomize parameters accordingly
        if isinstance(max_drift, tuple):
            self.drift_type = 'quadratic'
            self.a_x = np.random.uniform(-max_drift[0], max_drift[0])
            self.a_y = np.random.uniform(-max_drift[0], max_drift[0])
            self.b_x = np.random.uniform(-max_drift[1], max_drift[1])
            self.b_y = np.random.uniform(-max_drift[1], max_drift[1])
        else:
            self.drift_type = 'linear'
            self.m_x = np.random.uniform(-max_drift, max_drift)
            self.m_y = np.random.uniform(-max_drift, max_drift)

    def generate_gaussian_blob(self, x, y, x0, y0, sigma, amplitude):
        """
        Generate a 2D Gaussian blob at a given location with specified parameters.

        Parameters:
        - x, y: Meshgrid for the image dimensions.
        - x0, y0: Center of the gaussian blob.
        - sigma: Standard deviation of the gaussian blob.
        - amplitude: Amplitude of the gaussian blob (can be negative for valleys).

        Returns:
        - 2D numpy array representing the gaussian blob.
        """
        return amplitude * np.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))

    def generate_base_image(self, padding=0):
        """
        Generate a base image consisting of a noise floor and random gaussian blobs.

        Parameters:
        - padding: Additional space around the image to account for movement.

        Returns:
        - Padded base image with noise and gaussian blobs.
        """
        X_padded = self.X + 2 * padding
        Y_padded = self.Y + 2 * padding

        # Create a noise floor using random values
        noise_floor = self.background_noise * np.random.randn(X_padded, Y_padded)

        x = np.arange(X_padded)
        y = np.arange(Y_padded)
        x, y = np.meshgrid(x, y)

        # Add random gaussian blobs (peaks or valleys) to the noise floor
        for _ in range(self.n_blobs):
            x0 = np.random.randint(X_padded)
            y0 = np.random.randint(Y_padded)
            sigma = np.random.uniform(5, 15)
            if np.random.rand() < 0.5:
                amplitude = np.random.uniform(1, self.blob_amplitude)
            else:
                amplitude = np.random.uniform(-1, -self.blob_amplitude)
            noise_floor += self.generate_gaussian_blob(x, y, x0, y0, sigma, amplitude)
        return noise_floor

    def calculate_padding(self):
        """
        Calculate the necessary padding around the image to ensure that the entire
        field of view stays within the base image, even with maximum drift and jitter.

        Returns:
        - Amount of padding needed.
        """
        if self.drift_type == 'linear':
            max_drift_x = self.m_x * self.frames
            max_drift_y = self.m_y * self.frames
        else:
            max_drift_x = self.a_x * self.frames**2 + self.b_x * self.frames
            max_drift_y = self.a_y * self.frames**2 + self.b_y * self.frames
        total_max_shift_x = max_drift_x + self.frames * self.max_jitter
        total_max_shift_y = max_drift_y + self.frames * self.max_jitter
        return int(max(total_max_shift_x, total_max_shift_y))

    def simulate(self):
        """
        Simulate a sequence of images with drift and jitter.

        Returns:
        - Simulated data and list of shifts for each frame.
        """
        padding = self.calculate_padding()
        large_base_image = self.generate_base_image(padding)
        self.data = np.zeros((self.frames, self.X, self.Y))
        shifts = []
        for t in tqdm(range(self.frames)):
            # Add random jitter for the current frame
            jitter_x = np.random.uniform(-self.max_jitter, self.max_jitter)
            jitter_y = np.random.uniform(-self.max_jitter, self.max_jitter)

            # Calculate the drift for the current frame (linear or quadratic)
            if self.drift_type == 'linear':
                drift_x = self.m_x * t
                drift_y = self.m_y * t
            else:
                drift_x = self.a_x * t**2 + self.b_x * t
                drift_y = self.a_y * t**2 + self.b_y * t

            # Combine the drift and jitter to get the total shift for the current frame
            shift_x = jitter_x + drift_x
            shift_y = jitter_y + drift_y
            shifts.append((shift_x, shift_y))

            # Apply the shift using an affine transformation
            transformation_matrix = np.array([[1, 0, -shift_x], [0, 1, -shift_y], [0, 0, 1]])
            shifted_image = affine_transform(large_base_image, transformation_matrix[:2, :2],
                                             offset=transformation_matrix[:2, 2], mode='nearest', order=1)

            # Add frame-by-frame shot noise to the image
            shifted_image += self.shot_noise * np.random.randn(self.X + 2 * padding, self.Y + 2 * padding)

            # Extract the relevant section of the shifted image for the current frame
            start_x = padding
            end_x = start_x + self.X
            start_y = padding
            end_y = start_y + self.Y
            self.data[t] = shifted_image[start_x:end_x, start_y:end_y]
        self.shifts = shifts
        return self.data, shifts

    def save(self, path):
        """
        Save the simulated data to a .tiff file.

        Parameters:
        - path: Destination path for the .tiff file.
        """
        path = Path(path)
        imsave(path, self.data.astype(np.float32))

    def plot_overview(self, savefig=None):
        """
        Plot an overview of the simulated data, showing the first, middle, and last frames,
        as well as the shifts in the x and y directions over time.
        """
        shifts_x = [s[0] for s in self.shifts]
        shifts_y = [s[1] for s in self.shifts]

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        axs[0].imshow(self.data[0], cmap='gray')
        axs[0].set_title('First Frame')

        axs[1].imshow(self.data[self.frames//2], cmap='gray')
        axs[1].set_title('Middle Frame')

        axs[2].imshow(self.data[-1], cmap='gray')
        axs[2].set_title('Last Frame')

        axs[3].plot(shifts_x, label='Shift X')
        axs[3].plot(shifts_y, label='Shift Y')
        axs[3].set_title('Shifts over Time')
        axs[3].set_xlabel('Frame')
        axs[3].legend()

        for ax in axs[:3]:
            ax.axis('off')

        plt.tight_layout()

        if savefig is None:
            plt.show()
        else:
            fig.savefig(savefig)

if __name__ == "__main__":

    save_path = "../test/test.png"

    sim_data_obj_new = SimData(frames=250, X=50, Y=50, n_blobs=10, noise_amplitude=0.2,
                           blob_amplitude=5, max_drift=(0.0001, 0.01), max_jitter=1,
                           background_noise=1, shot_noise=0.2)

    data_new, shifts_new = sim_data_obj_new.simulate()
    sim_data_obj_new.plot_overview(savefig=save_path)
