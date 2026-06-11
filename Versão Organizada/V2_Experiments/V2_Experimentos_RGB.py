import rgb_noises as rn
import matplotlib.pyplot as plt
from skimage import data
import numpy as np
import lib_bp_img_3d_ord as bp
import pandas as pd
import os

dx = 2
dy = 2
D = dx * dy

# Dimensoes das imagens
width, height = 512, 512

def calc_entropy_complexity(img_rgb):
    probs = bp.bandt_pompe_method(img_rgb, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    return ent, comp

NOISE_EXPERIMENTS = {
    "Chroma Noise": {
        "experiment": True,
        "levels": [0.0, 0.05, 0.11, 0.17, 0.23, 0.29, 0.36, 0.42, 0.48, 0.54, 0.60], # Nova faixa de 10 níveis
        "func": lambda img, l: rn.add_chroma_noise(img, noise_strength=l, correlation=0.3)
    },
    "Color Misalignment": {
        "experiment": True,
        "levels": [0.0, 0.5, 1.1, 1.7, 2.3, 2.9, 3.6, 4.2, 4.8, 5.4, 6.0], # Nova faixa de 10 níveis
        "func": lambda img, l: rn.add_color_misalignment(img, shift_range=(0, l))
    },
    "Color Moiré": {
        "experiment": True,
        "levels": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # Nova faixa de 10 níveis
        "func": lambda img, l: rn.add_color_moire(img, amplitude_range=(l, l + 1e-6))
    },
    "Bayer Noise": {
        "experiment": False,
        "levels": [0.0, 0.01, 0.02, 0.04],
        "func": lambda img, l: rn.simulate_bayer_noise(img, (l, l/2, 1.2*l))
    },
    "Color-Dependent Noise": {
        "experiment": False,
        "levels": [0.0, 1.0, 1.5, 2.0],
        "func": lambda img, l: rn.add_color_dependent_noise(img, blue_noise_factor=l)
    },
    "White Balance Noise": {
        "experiment": False,
        "levels": [0.0, 500, 1000, 2000],
        "func": lambda img, l: rn.add_white_balance_noise(img, temperature_shift=l)
    },
    "JPEG Artifacts": {
        "experiment": True,
        "levels": [100, 95, 85, 70, 55, 40, 30, 20, 15, 10, 5], # Nova faixa de 10 níveis
        "func": lambda img, l: rn.add_jpeg_color_artifacts(img, quality=int(l))
    },
    "Chromatic Aberration": {
        "experiment": False,
        "levels": [0.0, 1.0, 2.0, 3.0],
        "func": lambda img, l: rn.add_chromatic_aberration(img, aberration_strength=l)
    },
    "Color Flare": {
        "experiment": False,
        "levels": [0.0, 0.15, 0.3, 0.5],
        "func": lambda img, l: rn.add_color_flare(img, flare_intensity=l)
    },
    "Demosaicing Artifacts": {
        "experiment": False,
        "levels": [0.0, 0.3, 0.6, 1.0],
        "func": lambda img, l: rn.add_demosaicing_artifacts(img, artifact_strength=l)
    }
}

# # Imagem 1 — Lena
# image_name = "Lena"
# image = data.astronaut().astype(np.uint8)

# # Imagem 2 — Rocket
# image_name = "Rocket"
# image = data.rocket().astype(np.uint8)

# Imagem 3 — Checkerboard colorido
image_name = "Checkerboard"
block_size = 32
image = np.zeros((height, width, 3), dtype=np.uint8)

color1 = [0, 0, 255]    # Azul
color2 = [255, 255, 0]  # Amarelo

for i in range(0, height, block_size):
    for j in range(0, width, block_size):
        if (i // block_size + j // block_size) % 2 == 0:
            image[i:i+block_size, j:j+block_size] = color1
        else:
            image[i:i+block_size, j:j+block_size] = color2

all_results = []
noise_dataframes = {}

NOISE_SELECTED = {
    k: v for k, v in NOISE_EXPERIMENTS.items()
    if v.get("experiment", False)
}

for noise_name, cfg in NOISE_SELECTED.items():
    
    rows = []
    
    for level in cfg["levels"]:
        if level == 0.0 or (noise_name == "JPEG Artifacts" and level == 100):
            img_proc = image
            intensity = 0.0
        else:
            img_proc = cfg["func"](image, level)
            intensity = level
        
        ent, comp = calc_entropy_complexity(img_proc)
        
        row = {
            "Imagem": image_name,
            "Tipo_Ruido": noise_name,
            "Intensidade": intensity,
            "Entropia": ent,
            "Complexidade": comp
        }
        
        rows.append(row)
        all_results.append(row)
    
    df_noise = pd.DataFrame(rows)
    noise_dataframes[noise_name] = df_noise

# DataFrame agregado
df_all = pd.DataFrame(all_results)

# Salvar em Excel
node = "C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/[NOVO] Versão Organizada/V2_Experiments/"
output_path = node + "resultados_" + image_name + "_ruidos_entropia_complexidade.xlsx"
df_all.to_excel(output_path, index=False, engine="openpyxl")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOT RESULTADOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%

PLOT_DIR = node + "plots_final/"

plt.figure(figsize=(8, 6))

# Ponto da imagem original (comum a todos)
df_orig = df_all[df_all["Intensidade"] == 0.0]
H0 = df_orig.iloc[0]["Entropia"]
C0 = df_orig.iloc[0]["Complexidade"]

plt.scatter(H0, C0, c="black", s=120, marker="*", label=image_name)

# Plotar cada ruído
for noise, df_noise in df_all.groupby("Tipo_Ruido"):
    df_sorted = df_noise.sort_values("Intensidade")
    
    plt.plot(
        df_sorted["Entropia"],
        df_sorted["Complexidade"],
        marker="o",
        linestyle="--",
        linewidth=1.5,
        label=noise
    )
    
    # ANOTAR APENAS PRIMEIRO E ÚLTIMO
    first = df_sorted.iloc[0]
    last = df_sorted.iloc[-1]
    
    plt.annotate(
        f"{first["Intensidade"]}",
        (first["Entropia"], first["Complexidade"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7
    )
    
    plt.annotate(
        f"{last["Intensidade"]}",
        (last["Entropia"], last["Complexidade"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7,
        fontweight="bold"
    )

plt.xlabel("Entropia")
plt.ylabel("Complexidade Estatística")
#plt.title("Entropy–Complexity Plane — Comparison Between Noise Types")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, image_name + "_mvacecp_trajectories_all_noises.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.savefig(
    os.path.join(PLOT_DIR, image_name + "_mvacecp_trajectories_all_noises.pdf"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#%% ####################  PLOT NO PLANO ###########################

fig, ax_main = plt.subplots(figsize=(9, 6))
bp.plot_3d(D)

# =========================
# DICIONÁRIOS DE TRADUÇÃO
# =========================
LEGEND_TRANSLATION = {
    "Chroma Noise": "Ruído de Crominância",
    "Color Misalignment": "Desalinhamento Cromático",
    "Color Moiré": "Moiré Cromático",
    "JPEG Artifacts": "Artefatos Cromáticos de JPEG"
}

IMAGE_LABEL_TRANSLATION = {
    "Lena": "Lena Astronauta",
    "Rocket": "Foguete",
    "Checkerboard": "Padrão Xadrez"
}

handles = []
labels = []

# =========================
# PLOT TRAJETÓRIAS (MAIN)
# =========================
for noise, df_noise in df_all.groupby("Tipo_Ruido"):
    df_sorted = df_noise.sort_values("Intensidade")
    
    noise_label_pt = LEGEND_TRANSLATION.get(noise, noise)

    line, = ax_main.plot(
        df_sorted["Entropia"],
        df_sorted["Complexidade"],
        marker="o",
        linestyle="--",
        linewidth=1.5,
        label=noise_label_pt,
        zorder=1
    )
    
    handles.append(line)
    labels.append(noise_label_pt)

    # --------- ANOTAÇÕES ----------
    if len(df_sorted) >= 2:
        second = df_sorted.iloc[1]
        last = df_sorted.iloc[-1]

        ax_main.annotate(
            f"{second["Intensidade"]}",
            (second["Entropia"], second["Complexidade"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            zorder=5
        )

        ax_main.annotate(
            f"{last["Intensidade"]}",
            (last["Entropia"], last["Complexidade"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            zorder=5
        )

# =========================
# IMAGEM ORIGINAL (MAIN)
# =========================
image_label_pt = IMAGE_LABEL_TRANSLATION.get(image_name, image_name)

star = ax_main.scatter(
    H0, C0,
    c="black",
    s=200,
    marker="*",
    edgecolor="white",
    linewidth=1.0,
    label=image_label_pt,
    zorder=10
)

handles = [star] + handles
labels = [image_label_pt] + labels

ax_main.set_xlabel("Entropia")
ax_main.set_ylabel("Complexidade Estatística")
ax_main.legend(handles, labels, fontsize=8, loc='upper left')
ax_main.grid(alpha=0.3)

plt.tight_layout()

# =========================
# INSET (ZOOM)
# =========================
axins = plt.axes([1.05, 0.27, 0.6, 0.6])

x1, x2, y1, y2 = 0.0, 0.45, 0.0, 0.35
#LENA = 0.7, 1.0, 0.0, 0.3
#ROCKET = 0.75, 1.0, 0.0, 0.25
#CHECKERBOARD = 0.0, 0.45, 0.0, 0.35
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

ax_main.indicate_inset_zoom(axins, edgecolor='0.4')

bp.plot_3d(D)

# =========================
# TRAJETÓRIAS NO INSET + RÓTULOS
# =========================
for noise, df_noise in df_all.groupby("Tipo_Ruido"):
    df_sorted = df_noise.sort_values("Intensidade")

    axins.plot(
        df_sorted["Entropia"],
        df_sorted["Complexidade"],
        marker="o",
        linestyle="--",
        linewidth=1.5,
        zorder=1
    )

    if len(df_sorted) >= 2:
        second = df_sorted.iloc[1]
        last = df_sorted.iloc[-1]

        axins.annotate(
            f"{second["Intensidade"]}",
            (second["Entropia"], second["Complexidade"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=6,
            zorder=5
        )

        axins.annotate(
            f"{last["Intensidade"]}",
            (last["Entropia"], last["Complexidade"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=6,
            fontweight="bold",
            zorder=5
        )

# Estrela no inset
axins.scatter(
    H0, C0,
    c="black",
    s=120,
    marker="*",
    edgecolor="white",
    linewidth=0.8,
    zorder=10
)

fig.savefig(
    os.path.join(PLOT_DIR, image_name + "_mvacecp_trajectories_with_inset_labeled.png"),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(PLOT_DIR, image_name + "_mvacecp_trajectories_with_inset_labeled.pdf"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()