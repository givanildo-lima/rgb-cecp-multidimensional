import rgb_noises as rn
import matplotlib.pyplot as plt
from skimage import data
import numpy as np
import os

# pasta de saída
os.makedirs("example_noise_results", exist_ok=True)

# carregar imagem do gato
#image = data.chelsea().astype(np.uint8)
image = data.astronaut().astype(np.uint8)

# --------------------------------------------------
# salvar imagem original
# --------------------------------------------------

plt.figure(figsize=(5,5))
plt.imshow(image)
#plt.title("Original Image")
plt.axis("off")
plt.tight_layout()
plt.savefig("example_noise_results/cat_original.png", dpi=300)
plt.close()


# --------------------------------------------------
# função para aplicar ruído e salvar resultado
# --------------------------------------------------

def apply_and_save_noise(image, noise_name, noise_func, level):
    # Aplica o ruído
    noisy = noise_func(image, level)
    
    # Configura a plotagem
    plt.figure(figsize=(5,5))
    plt.imshow(noisy)
    #plt.title(f"{noise_name} (Level: {level})")
    plt.axis("off")
    plt.tight_layout()
    
    # Formata o nome do arquivo: remove espaços, converte para minúsculas e inclui o nível
    # Substitui o ponto decimal por underline para evitar problemas de extensão
    clean_name = noise_name.replace(" ", "_").lower()
    level_str = str(level).replace(".", "_")
    filename = f"{clean_name}_level_{level_str}.png"
    
    # Salva o arquivo na pasta de resultados
    plt.savefig(f"example_noise_results/{filename}", dpi=300)
    plt.close()

# --------------------------------------------------
# wrappers de ruído
# --------------------------------------------------

def chroma_noise(img, level):
    return rn.add_chroma_noise(img, noise_strength=level, correlation=0.3)

def misalignment_noise(img, level):
    return rn.add_color_misalignment(img, shift_range=(0, level))

def moire_noise(img, level):
    return rn.add_color_moire(img, amplitude_range=(level, level+1e-6))

def jpeg_noise(img, level):
    return rn.add_jpeg_color_artifacts(img, quality=int(level))


# --------------------------------------------------
# aplicar ruídos (intensidade alta)
# --------------------------------------------------

apply_and_save_noise(image, "Chroma Noise", chroma_noise, 0.1)
apply_and_save_noise(image, "Color Misalignment", misalignment_noise, 2.0)
apply_and_save_noise(image, "Color Moiré", moire_noise, 0.3)
apply_and_save_noise(image, "JPEG Artifacts", jpeg_noise, 20)

apply_and_save_noise(image, "Chroma Noise", chroma_noise, 0.5)
apply_and_save_noise(image, "Color Misalignment", misalignment_noise, 5.0)
apply_and_save_noise(image, "Color Moiré", moire_noise, 0.8)
apply_and_save_noise(image, "JPEG Artifacts", jpeg_noise, 5)