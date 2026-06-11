import numpy as np

# Dimensoes das imagens
width, height = 512, 512

def add_gaussian_noise_rgb(image=np.ones((height, width, 3)), mean=0, factor=0.1):
    '''
    Adiciona ruído gaussiano a uma imagem RGB (colorida).
    '''
    if not isinstance(image, np.ndarray):
        raise TypeError("A imagem deve ser um numpy array")
    
    if image.dtype != np.float64:
        raise ValueError("A imagem precisa ter dtype float64")
    
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Os valores devem estar no intervalo [0, 1]")

    # Verifica se é uma imagem RGB
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A imagem deve ser RGB (altura, largura, 3)")

    # Special case: if factor=0, return original image without noise
    if factor == 0:
        return image.copy()         
    
    noisy_image = np.zeros_like(image)
    for channel in range(3):  # para cada canal R, G, B
        gauss = np.random.normal(mean, factor, image[:, :, channel].shape)
        noisy_channel = image[:, :, channel] + gauss
        
        # noisy_channel = np.clip(noisy_channel, 0, 1)  # mantêm valores entre [0, 1]
        # noisy_image[:, :, channel] = noisy_channel
        
        # extracts the min and max of the noised image
        noisy_min = noisy_channel.min()
        noisy_max = noisy_channel.max()    
        # normalize to [0, 1]
        normalized = (noisy_channel - noisy_min) / (noisy_max - noisy_min)
        
        noisy_image[:, :, channel] = normalized
    
    return noisy_image

###############################################################################

def add_speckle_noise_rgb(image=np.ones((height, width, 3)), mean=0, factor=0.2):
    if not isinstance(image, np.ndarray):
        raise TypeError("A imagem deve ser um numpy array")

    if image.dtype != np.float64:
        raise ValueError("A imagem precisa ter dtype float64")

    if image.min() < 0 or image.max() > 1:
        raise ValueError("Os valores devem estar no intervalo [0, 1]")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A imagem deve ser RGB (altura, largura, 3)")
    
    # Special case: if factor=0, return original image without noise
    if factor == 0:
        return image.copy()

    noisy_image = np.zeros_like(image)
    for c in range(3):
        gauss = np.random.normal(mean, factor, image[:, :, c].shape)
        noisy_channel = image[:, :, c] + image[:, :, c] * gauss
        
        # noisy_channel = np.clip(noisy_channel, 0, 1)
        # noisy_image[:, :, c] = noisy_channel        
        
        # extracts the min and max of the noised image
        noisy_min = noisy_channel.min()
        noisy_max = noisy_channel.max()    
        # normalize to [0, 1]
        normalized = (noisy_channel - noisy_min) / (noisy_max - noisy_min)
        
        noisy_image[:, :, c] = normalized

    return noisy_image

###############################################################################

def add_poisson_noise_rgb(image=np.ones((height, width, 3)), factor=1.0):
    """
    Add poisson noise to an RGB image.

    Parameters:
    image (numpy.ndarray(np.float64)): Input RGB image (Intensities in range [0, 1])
    factor (float): Scaling factor to control noise intensity. Higher values produce more noise.
                    Factor=0 produces no noise. (default 1.0)

    Returns:
    numpy.ndarray(np.float64): RGB Image with added poisson noise
    """
    
    # Special case: if factor=0, return original image without noise
    if factor == 0.0:
        return image.copy()
    
    # Check if input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")

    # Check if dtype is float64
    if image.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")

    # Check if pixel values are in range [0, 1]
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")

    # Check if image is RGB
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB (shape [height, width, 3])")

    noisy_image = np.zeros_like(image)

    # Apply Poisson noise independently to each channel
    for channel in range(3):
        noisy_channel = np.random.poisson(image[:, :, channel] * 255.0 / factor) * factor / 255.0
        
        #noisy_image[:, :, channel] = np.clip(noisy_channel, 0.0, 1.0)
        
        # extracts the min and max of the noised image
        noisy_min = noisy_channel.min()
        noisy_max = noisy_channel.max()    
        # normalize to [0, 1]
        normalized = (noisy_channel - noisy_min) / (noisy_max - noisy_min)
        
        noisy_image[:, :, channel] = normalized

    return noisy_image

###############################################################################

def add_sp_noise_rgb(image=np.ones((height, width, 3)), factor=0.3):
    
    salt_prob=factor
    pepper_prob=factor
    
    """
    Add salt and pepper noise to an RGB image.

    Parameters:
    image (numpy.ndarray(np.float64)): RGB input image (Intensities in range [0, 1])
    salt_prob (float): Probability of salt noise (white pixels), range [0, 1]
    pepper_prob (float): Probability of pepper noise (black pixels), range [0, 1]

    Returns:
    numpy.ndarray(np.float64): RGB image with added salt and pepper noise
    """
    # Check if input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")

    # Check if dtype is float64
    if image.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")

    # Check if pixel values are in range [0, 1]
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")
    
    # Special case: if factor=0, return original image without noise
    if factor == 0:
        return image.copy()

    # Validate probabilities
    if not (0 <= salt_prob <= 1 and 0 <= pepper_prob <= 1):
        raise ValueError("Probabilities must be between 0 and 1")
    if salt_prob + pepper_prob > 1:
        raise ValueError("Sum of probabilities must not exceed 1")

    # Create copy of image
    noisy = image.copy()

    # Generate random noise mask (same for all channels)
    mask = np.random.random(image.shape[:2])

    # Add salt noise (white pixels)
    salt_mask = mask < salt_prob
    noisy[salt_mask, :] = 1

    # Add pepper noise (black pixels)
    pepper_mask = (mask >= salt_prob) & (mask < salt_prob + pepper_prob)
    noisy[pepper_mask, :] = 0

    return noisy

############################### RUÍDOS MÁXIMOS ################################

height = 256
width = 256

gauss_max = add_gaussian_noise_rgb(
    np.zeros((height, width, 3), dtype=np.float64),
    mean=0,
    factor=5
)

speckle_max = add_speckle_noise_rgb(
    np.ones((height, width, 3), dtype=np.float64),
    mean=0,
    factor=3
)

poisson_max = add_poisson_noise_rgb(
    np.ones((height, width, 3), dtype=np.float64),
    factor=0.1
)

sp_max = add_sp_noise_rgb(
    np.zeros((height, width, 3), dtype=np.float64),
    factor=0.5
)

####################################### TESTE #################################

# from skimage import data
# import matplotlib.pyplot as plt

# lena_image = data.astronaut()/255.0

# # Adiciona ruído gaussiano
# noisy_image = add_poisson_noise_rgb(lena_image, factor = 1.0)

# # Exibe a imagem original e a com ruído lado a lado
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(lena_image)
# ax[0].set_title("Imagem Original")
# ax[0].axis("off")

# ax[1].imshow(noisy_image)
# ax[1].set_title("Imagem com Ruído")
# ax[1].axis("off")

# plt.tight_layout()
# plt.show()