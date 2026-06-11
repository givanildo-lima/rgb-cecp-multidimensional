import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import pandas as pd
import cv2
import os

import lib_bp_img_3d_ord as bp
import vf_ruidos as noises
import gerar_imagens as dataset
import lib_ribeiro as ordpy

# %% ==============================
# PARÂMETROS GERAIS
# ==============================

dx, dy = 2, 2  # Dimensões de embedding
D = dx * dy
np.random.seed(42)  # Reprodutibilidade
os.makedirs("./plots/imgs/", exist_ok=True)

# %% ==============================
# RUÍDOS MÁXIMOS
# ==============================
height = 256
width = 256

gauss_max = noises.add_gaussian_noise_rgb(
    np.zeros((height, width, 3), dtype=np.float64),
    mean=0,
    factor=5
)

speckle_max = noises.add_speckle_noise_rgb(
    np.ones((height, width, 3), dtype=np.float64),
    mean=0,
    factor=3
)

poisson_max = noises.add_poisson_noise_rgb(
    np.ones((height, width, 3), dtype=np.float64),
    factor=0.1
)

sp_max = noises.add_sp_noise_rgb(
    np.zeros((height, width, 3), dtype=np.float64),
    factor=0.5
)

# %% ==============================
# FUNÇÕES AUXILIARES
# ==============================

def calc_entropy_complexity(img_rgb, img_name, noise_type, level):
    """Calcula entropia e complexidade para uma imagem RGB."""
    probs = bp.bandt_pompe_method(img_rgb, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    return [img_name, noise_type, level, ent, comp]


def apply_noise_per_image(img, img_name, noise_levels):
    """Aplica todos os tipos de ruído em uma única imagem."""
    results = []

    for noise_type, levels in noise_levels.items():
        for level in levels:
            if noise_type == "gaussian":
                noisy = noises.add_gaussian_noise_rgb(img, factor=level)
            elif noise_type == "speckle":
                noisy = noises.add_speckle_noise_rgb(img, factor=level)
            elif noise_type == "poisson":
                noisy = noises.add_poisson_noise_rgb(img, factor=level)
            elif noise_type == "sp":
                noisy = noises.add_sp_noise_rgb(img, factor=level)
            else:
                continue

            results.append(calc_entropy_complexity(noisy, img_name, noise_type, level))
    print(results)
    return results


def get_zoom_limits_by_image(img_name):
    """Define limites de zoom personalizados por imagem."""
    zoom_config = {
        "Lena": (0.90, 1.00, 0.00, 0.12),
        #"Cat": (0.88, 1.00, 0.00, 0.15),
        #"Coffee": (0.89, 1.00, 0.00, 0.10),
        "Rocket": (0.85, 0.99, 0.00, 0.16),
        #"Uniform": (0.96, 1.00, 0.00, 0.05),
        "Chess": (0.92, 1.00, 0.00, 0.10),
    }
    return zoom_config.get(img_name, (0.9, 1.0, 0.0, 0.12))


def plot_noise_comparison(df, D, img_name, colors_noise):
    """Gera gráfico comparativo dos ruídos aplicados a uma única imagem."""
    fig, ax_main = plt.subplots(figsize=(9, 6))
    bp.plot_3d(D)

    # Curvas dos diferentes ruídos
    for noise_type, color in colors_noise.items():
        subset = df[df["Tipo_Ruido"] == noise_type]
        ax_main.plot(
            subset["Entropia"],
            subset["Complexidade"],
            marker="o",
            linestyle="--",
            label=noise_type.capitalize(),
            color=color
        )
        
    ent_img = df_img.iloc[0]["Entropia"]
    comp_img = df_img.iloc[0]["Complexidade"]
    label_img = df_img.iloc[0]["Imagem"]
    
    ax_main.plot(
        ent_img,
        comp_img,
        marker="o",
        linestyle="--",
        label=label_img,
        color='cyan'
    )

    ax_main.set_title(f"MvCECP - {img_name} com diferentes ruídos", fontsize=14)
    ax_main.set_xlabel("Entropy", fontsize=12)
    ax_main.set_ylabel("Complexity", fontsize=12)
    ax_main.legend(title="Tipo de Ruído", loc="upper left", fontsize=9)
    plt.tight_layout()

    # Zoom dinâmico
    zoom_area = get_zoom_limits_by_image(img_name)
    axins = plt.axes([1.05, 0.27, 0.6, 0.6])
    x1, x2, y1, y2 = zoom_area
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    ax_main.indicate_inset_zoom(axins, edgecolor="0.4")

    # Replota dados dentro do zoom
    for noise_type, color in colors_noise.items():
        subset = df[df["Tipo_Ruido"] == noise_type]
        axins.plot(
            subset["Entropia"],
            subset["Complexidade"],
            marker="o",
            linestyle="--",
            color=color
        )
    
    axins.plot(
        ent_img,
        comp_img,
        marker="o",
        linestyle="--",
        label=label_img,
        color='cyan'
    )

    # Adiciona limites teóricos do plano CECP
    folder = './limits/mv/'
    M = 240
    df_cont = pd.read_csv(os.path.join(folder, f'continua-N{M}.q1'),
                          skiprows=20, sep='  ', engine='python', names=['HT', 'CJT'])
    df_troz = pd.read_csv(os.path.join(folder, f'trozos-N{M}.q1'),
                          skiprows=20, sep='  ', engine='python', names=['HT', 'CJT'])
    axins.plot(df_cont['HT'], df_cont['CJT'], color='black')
    axins.plot(df_troz['HT'], df_troz['CJT'], color='black')

    # Salva e mostra
    #output_path = f"./plots/imgs/plot_{img_name.lower()}_ruidos.png"
    output_path = f"C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/plots/imgs/plot_{img_name.lower()}_ruidos.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    output_path = f"C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/plots/imgs/plot_{img_name.lower()}_ruidos.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Gráfico salvo em: {output_path}")
    plt.show()


# %% ==============================
# IMAGENS BASE
# ==============================

imgs_dict = {
    "Lena": data.astronaut().astype(np.float64)/255.0,
    #"Cat": data.cat().astype(np.float64)/255.0,
    #"Coffee": data.coffee().astype(np.float64)/255.0,
    "Rocket": data.rocket().astype(np.float64)/255.0,
    #"Uniform": dataset.uniform_image.astype(np.float64),
    "Chess": dataset.chessboard_rgb.astype(np.float64),
}

# Cores para cada tipo de ruído
colors_noise = {
    "gaussian": "orange",
    "speckle": "blue",
    "poisson": "green",
    "sp": "red",
}

# %% ==============================
# CONFIGURAÇÃO DE NÍVEIS DE RUÍDO
# ==============================

noise_levels = {
    "gaussian": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "speckle": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "poisson": [0.0, 0.25, 0.5, 0.75, 1.0],
    "sp": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}

# %% ==============================
# LOOP AUTOMÁTICO POR IMAGEM
# ==============================

for img_name, img_data in imgs_dict.items():
    print(f"\n=== Analisando imagem: {img_name.upper()} ===")
    results = apply_noise_per_image(img_data, img_name, noise_levels)

    df_img = pd.DataFrame(
        results,
        columns=["Imagem", "Tipo_Ruido", "Intensidade", "Entropia", "Complexidade"]
    )
    
    plot_noise_comparison(df_img, D, img_name, colors_noise)
    
