import rgb_noises as rn
import matplotlib.pyplot as plt
from skimage import data
import numpy as np

#%% ################### PLOTANDO RUÍDOS ###########################

INTENSITY_LEVELS = {
    "chroma": [0.0, 0.02, 0.05, 0.1],
    "misalignment": [0.0, 0.5, 1.0, 2.0],
    "moire": [0.0, 0.1, 0.2, 0.3],
    "bayer": [0.0, 0.01, 0.02, 0.04],
    "color_dependent": [0.0, 1.0, 1.5, 2.0],
    "white_balance": [0.0, 500, 1000, 2000],
    "crosstalk": [0.0, 0.1, 0.2, 0.3],
    "jpeg": [100, 70, 40, 20],
    "chromatic_aberration": [0.0, 1.0, 2.0, 3.0],
    "flare": [0.0, 0.15, 0.3, 0.5],
    "demosaic": [0.0, 0.3, 0.6, 1.0]
}

def plot_noise_progression(image, noise_name, noise_func, levels, titlesuffix=""):
    """
    Gera um print com a degradação progressiva da imagem para um tipo de ruído.
    """
    n = len(levels)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    for i, level in enumerate(levels):
        if level == 0.0:
            img = image
            title = "Original"
        else:
            img = noise_func(image, level)
            title = f"{titlesuffix}{level}"
        
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis("off")
    
    plt.suptitle(f"Degradação progressiva — {noise_name}", fontsize=14)
    plt.tight_layout()
    plt.show()

def chroma_wrapper(img, level):
    return rn.add_chroma_noise(img, noise_strength=level, correlation=0.3)

def misalignment_wrapper(img, level):
    return rn.add_color_misalignment(img, shift_range=(0, level))

def moire_wrapper(img, level):
    return rn.add_color_moire(img, amplitude_range=(level, level+1e-6))

def bayer_wrapper(img, level):
    return rn.simulate_bayer_noise(img, (level, level/2, level*1.2))

def color_dependent_wrapper(img, level):
    return rn.add_color_dependent_noise(img, blue_noise_factor=level)

def white_balance_wrapper(img, level):
    return rn.add_white_balance_noise(img, temperature_shift=level)

def jpeg_wrapper(img, level):
    return rn.add_jpeg_color_artifacts(img, quality=int(level))

def chromatic_aberration_wrapper(img, level):
    return rn.add_chromatic_aberration(img, aberration_strength=level)

def flare_wrapper(img, level):
    return rn.add_color_flare(img, flare_intensity=level)

def demosaic_wrapper(img, level):
    return rn.add_demosaicing_artifacts(img, artifact_strength=level)

image = data.astronaut().astype(np.uint8)

plot_noise_progression(image, "Chroma Noise", chroma_wrapper, [0.0, 0.02, 0.05, 0.1])
plot_noise_progression(image, "Color Misalignment", misalignment_wrapper, [0.0, 0.5, 1.0, 2.0])
plot_noise_progression(image, "Color Moiré", moire_wrapper, [0.0, 0.1, 0.2, 0.3])
plot_noise_progression(image, "Bayer Noise", bayer_wrapper, [0.0, 0.01, 0.02, 0.04])
plot_noise_progression(image, "Color-Dependent Noise", color_dependent_wrapper, [0.0, 1.0, 1.5, 2.0])
plot_noise_progression(image, "White Balance Noise", white_balance_wrapper, [0.0, 500, 1000, 2000])
plot_noise_progression(image, "JPEG Artifacts", jpeg_wrapper, [100, 70, 40, 20])
plot_noise_progression(image, "Chromatic Aberration", chromatic_aberration_wrapper, [0.0, 1.0, 2.0, 3.0])
plot_noise_progression(image, "Color Flare", flare_wrapper, [0.0, 0.15, 0.3, 0.5])
plot_noise_progression(image, "Demosaicing Artifacts", demosaic_wrapper, [0.0, 0.3, 0.6, 1.0])
