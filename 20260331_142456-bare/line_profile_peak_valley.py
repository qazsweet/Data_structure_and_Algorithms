import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

profile_folder = "line_profile"

# List all npy files that match pattern profile_1_*.npy
profile_files = [f for f in os.listdir(profile_folder) if f.startswith('profile_1_') and f.endswith('.npy')]
profile_files.sort()  # optional, for deterministic order

for fname in profile_files:
    fpath = os.path.join(profile_folder, fname)
    profile = np.load(fpath)
    x = np.arange(len(profile))

    # Find peaks (maxima)
    peaks, _ = find_peaks(profile)
    # Find valleys (minima)
    valleys, _ = find_peaks(-profile)

    print(f'File: {fname}')
    print(f'  Peaks at: {peaks}, values: {profile[peaks]}' )
    print(f'  Valleys at: {valleys}, values: {profile[valleys]}' )

    # (Optional) plot profile with peaks and valleys
    plt.figure(figsize=(10,4))
    plt.plot(x, profile, label='Profile')
    plt.plot(peaks, profile[peaks], 'ro', label='Peaks')
    plt.plot(valleys, profile[valleys], 'go', label='Valleys')
    plt.title(f'Profile: {fname}')
    plt.legend()
    plt.show()