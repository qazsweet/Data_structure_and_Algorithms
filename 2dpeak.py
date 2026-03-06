import numpy as np
from scipy.optimize import curve_fit

def gaussian_2d(coords, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """Defined 2D Gaussian function for fitting."""
    x, y = coords
    # Robust implementation to avoid exponential overflow
    g = offset + amplitude * np.exp(
        -(((x - xo)**2 / (2 * sigma_x**2)) + ((y - yo)**2 / (2 * sigma_y**2)))
    )
    return g.ravel()

def fit_subpixel_peak(image_data, initial_guess_coord, window_size=3):
    """
    Fits a 2D Gaussian to a local window to find sub-pixel peak.
    
    :param image_data: 2D numpy array (the image)
    :param initial_guess_coord: (y, x) of the pixel-level maximum
    :param window_size: Radius of the window (e.g., 3 means a 7x7 grid)
    :return: (fine_y, fine_x), sigma, amplitude
    """
    y0, x0 = initial_guess_coord
    
    # 1. Extract local window
    y_min, y_max = max(0, y0 - window_size), min(image_data.shape[0], y0 + window_size + 1)
    x_min, x_max = max(0, x0 - window_size), min(image_data.shape[1], x0 + window_size + 1)
    
    sub_image = image_data[y_min:y_max, x_min:x_max]
    
    # 2. Create local grid
    y_range = np.arange(y_min, y_max)
    x_range = np.arange(x_min, x_max)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    
    # 3. Initial Parameters [amplitude, xo, yo, sigma_x, sigma_y, offset]
    amp_init = np.max(sub_image) - np.min(sub_image)
    initial_p = [amp_init, x0, y0, 1.0, 1.0, np.min(sub_image)]
    
    # 4. Perform Fit
    try:
        popt, _ = curve_fit(gaussian_2d, (x_grid, y_grid), sub_image.ravel(), p0=initial_p)
        return popt[2], popt[1], (popt[3], popt[4]), popt[0] # fine_y, fine_x, sigmas, amp
    except Exception as e:
        print(f"Fit failed: {e}")
        return float(y0), float(x0), None, None

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy image with a spot at (15.34, 12.78)
    img = np.random.normal(0, 0.1, (30, 30))
    y_true, x_true = 15.34, 12.78
    yy, xx = np.indices(img.shape)
    img += 10 * np.exp(-(((xx - x_true)**2 + (yy - y_true)**2) / (2 * 1.2**2)))

    # Pixel-level detection
    peak_y, peak_x = np.unravel_index(np.argmax(img), img.shape)
    
    # Sub-pixel fitting
    fine_y, fine_x, sigmas, amp = fit_subpixel_peak(img, (peak_y, peak_x))
    
    print(f"Pixel-level Peak: ({peak_y}, {peak_x})")
    print(f"Sub-pixel Fitted: ({fine_y:.4f}, {fine_x:.4f})")
    print(f"Ground Truth:    ({y_true}, {x_true})")
    print(f"Error:           ({abs(fine_y-y_true):.4f}, {abs(fine_x-x_true):.4f}) pixels")
