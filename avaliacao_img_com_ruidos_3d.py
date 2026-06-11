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

# Parametros para o calculo de entropia e complexidade
dx, dy = 2, 2 # Dimensoes de embedding
D = dx * dy

# Definir a semente global para reprodutibilidade
np.random.seed(42)

# Criar pasta de saída para os plots
os.makedirs("./plots", exist_ok=True)

# %% ==============================
# FUNÇÕES AUXILIARES
# ==============================

def calc_entropy_complexity(img_rgb, name, noise_type, level):
    """Calcula entropia e complexidade para uma imagem RGB."""
    probs = bp.bandt_pompe_method(img_rgb, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    return [name, noise_type, level, ent, comp]

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
# IMAGENS BASE
# ==============================

######## LENA ASTROUNATA ########

print("LENA\n")
lena_3d = data.astronaut().astype(np.float64)/255.0

probs_lena_3d = bp.bandt_pompe_method(lena_3d, dx, dy)
ent_lena_3d = bp.shannon_entropy(probs_lena_3d, normalized=True, sigma = D)
comp_lena_3d = bp.complexity(probs_lena_3d, ent_lena_3d)

lena_2d = cv2.cvtColor(lena_3d.astype(np.float32), cv2.COLOR_BGR2GRAY)

h_lena_2d, c_lena_2d = ordpy.get_entropy_complexity(lena_2d, dx=dx, dy=dy, taux=1, tauy=1)

#plt.imshow(lena_3d)
#plt.imshow(lena_2d, cmap='gray')

######## GATINHO ########

print("CAT\n")
cat_3d = data.cat().astype(np.float64)/255.0

probs_cat_3d = bp.bandt_pompe_method(cat_3d, dx, dy)
ent_cat_3d = bp.shannon_entropy(probs_cat_3d, normalized=True, sigma = D)
comp_cat_3d = bp.complexity(probs_cat_3d, ent_cat_3d)

cat_2d = cv2.cvtColor(cat_3d.astype(np.float32), cv2.COLOR_BGR2GRAY)

h_cat_2d, c_cat_2d = ordpy.get_entropy_complexity(cat_2d, dx=dx, dy=dy, taux=1, tauy=1)

# plt.imshow(cat_3d)
# plt.imshow(cat_2d, cmap='gray')

######## XÍCARA DE CAFÉ ########

print("COFFEE\n")
coffee_3d = data.coffee().astype(np.float64)/255.0

probs_coffee_3d = bp.bandt_pompe_method(coffee_3d, dx, dy)
ent_coffee_3d = bp.shannon_entropy(probs_coffee_3d, normalized=True, sigma = D)
comp_coffee_3d = bp.complexity(probs_coffee_3d, ent_coffee_3d)

coffee_2d = cv2.cvtColor(coffee_3d.astype(np.float32), cv2.COLOR_BGR2GRAY)

h_coffee_2d, c_coffee_2d = ordpy.get_entropy_complexity(coffee_2d, dx=dx, dy=dy, taux=1, tauy=1)

# plt.imshow(coffee_3d)
# plt.imshow(coffee_2d, cmap='gray')

######## FOGUETE ########

print("ROCKET\n")
rocket_3d = data.rocket().astype(np.float64)/255.0

probs_rocket_3d = bp.bandt_pompe_method(rocket_3d, dx, dy)
ent_rocket_3d = bp.shannon_entropy(probs_rocket_3d, normalized=True, sigma = D)
comp_rocket_3d = bp.complexity(probs_rocket_3d, ent_rocket_3d)

rocket_2d = cv2.cvtColor(rocket_3d.astype(np.float32), cv2.COLOR_BGR2GRAY)

h_rocket_2d, c_rocket_2d = ordpy.get_entropy_complexity(rocket_2d, dx=dx, dy=dy, taux=1, tauy=1)

# plt.imshow(rocket_3d)
# plt.imshow(rocket_2d, cmap='gray')

######## IMAGEM UNIFORME ########

print("UNIFORM\n")
uniform_3d = dataset.uniform_image.astype(np.float64)

probs_uniform_3d = bp.bandt_pompe_method(uniform_3d, dx, dy)
ent_uniform_3d = bp.shannon_entropy(probs_uniform_3d, normalized=True, sigma = D)
comp_uniform_3d = bp.complexity(probs_uniform_3d, ent_uniform_3d)

uniform_2d = cv2.cvtColor(uniform_3d.astype(np.float32), cv2.COLOR_BGR2GRAY)

h_uniform_2d, c_uniform_2d = ordpy.get_entropy_complexity(uniform_2d, dx=dx, dy=dy, taux=1, tauy=1)

# plt.imshow(uniform_3d)
# plt.imshow(uniform_2d, cmap='gray')

######## XADREZ COLORIDO ########

print("CHESS\n")
chess_3d = dataset.chessboard_rgb.astype(np.float64)

probs_chess_3d = bp.bandt_pompe_method(chess_3d, dx, dy)
ent_chess_3d = bp.shannon_entropy(probs_chess_3d, normalized=True, sigma = D)
comp_chess_3d = bp.complexity(probs_chess_3d, ent_chess_3d)

chess_2d = cv2.cvtColor(chess_3d.astype(np.float32), cv2.COLOR_BGR2GRAY)

h_chess_2d, c_chess_2d = ordpy.get_entropy_complexity(chess_2d, dx=dx, dy=dy, taux=1, tauy=1)

# plt.imshow(chess_3d)
# plt.imshow(chess_2d, cmap='gray')

# %% ################ PLOTS DE ENTROPIA E COMPLEXIDADE 3D ##################

fig, ax_main = plt.subplots(figsize=(10, 6))
    
# Adicionar limites no grafico principal
bp.plot_3d(D)

# Plot no grafico principal
ax_main.plot(ent_lena_3d, comp_lena_3d, marker='o', label='Lena Astronaut', color='blue')
ax_main.plot(ent_cat_3d, comp_cat_3d, marker='o', label="Cat", color='green')
ax_main.plot(ent_coffee_3d, comp_coffee_3d, marker='o', label="Coffee", color='yellow')
ax_main.plot(ent_rocket_3d, comp_rocket_3d, marker='o', label="Rocket", color='pink')
ax_main.plot(ent_uniform_3d, comp_uniform_3d, marker='o', label="Uniform", color='gray')
ax_main.plot(ent_chess_3d, comp_chess_3d, marker='o', label="Chessboard", color='red')
#ax_main.plot(ent_colored_3d, comp_colored_3d, marker='o', label="Colored", color='purple')

# Configuracoes do grafico principal
ax_main.set_title("MvCECP", fontsize=14)
ax_main.set_xlabel("Entropia", fontsize=12)
ax_main.set_ylabel("Complexidade", fontsize=12)
ax_main.legend(loc="upper left", fontsize=10)

plt.show()

# %% ################# PLOTS DE ENTROPIA E COMPLEXIDADE 2D ##################

fig, ax_main = plt.subplots(figsize=(10, 6))
    
# Adicionar limites no grafico principal
bp.plot_2d(D)

# Plot no grafico principal
ax_main.plot(h_lena_2d, c_lena_2d, marker='o', label='Lena Astronaut', color='blue')
ax_main.plot(h_cat_2d, c_cat_2d, marker='o', label="Cat", color='green')
ax_main.plot(h_coffee_2d, c_coffee_2d, marker='o', label="Coffee", color='yellow')
ax_main.plot(h_rocket_2d, c_rocket_2d, marker='o', label="Rocket", color='pink')
ax_main.plot(h_uniform_2d, c_uniform_2d, marker='o', label="Uniform", color='gray')
ax_main.plot(h_chess_2d, c_chess_2d, marker='o', label="Chessboard", color='red')
#ax_main.plot(h_colored_2d, c_colored_2d, marker='o', label="Colored", color='purple')

# Configuracoes do grafico principal
ax_main.set_title("Ribeiro", fontsize=14)
ax_main.set_xlabel("Entropia", fontsize=12)
ax_main.set_ylabel("Complexidade", fontsize=12)
ax_main.legend(loc="upper left", fontsize=10)

plt.show()

# %% AVALIAÇÕES DOS RUÍDOS

noise_levels = {
    "gaussian": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "sp": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "poisson": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], 
    "speckle": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
}

# %% RUÍDO GAUSSIANO ##

gaussian_results = []

for level in noise_levels["gaussian"]:
    noisy = noises.add_gaussian_noise_rgb(lena_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    gaussian_results.append(["Lena", "Gaussian", level, ent, comp])
    
# for level in noise_levels["gaussian"]:
#     noisy = noises.add_gaussian_noise_rgb(cat_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     gaussian_results.append(["Cat", "Gaussian", level, ent, comp])
    
# for level in noise_levels["gaussian"]:
#     noisy = noises.add_gaussian_noise_rgb(coffee_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     gaussian_results.append(["Coffee", "Gaussian", level, ent, comp])
    
for level in noise_levels["gaussian"]:
    noisy = noises.add_gaussian_noise_rgb(rocket_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    gaussian_results.append(["Rocket", "Gaussian", level, ent, comp])
    
# for level in noise_levels["gaussian"]:
#     noisy = noises.add_gaussian_noise_rgb(uniform_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     gaussian_results.append(["Uniform", "Gaussian", level, ent, comp])
    
for level in noise_levels["gaussian"]:
    noisy = noises.add_gaussian_noise_rgb(chess_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    gaussian_results.append(["Chess", "Gaussian", level, ent, comp])

# df_noises = pd.DataFrame(
#     gaussian_results,
#     columns=["Imagem", "Tipo_Ruido", "Intensidade", "Entropia", "Complexidade"]
# )

# %% RUÍDO SALT AND PEPPER ##

sp_results = []

for level in noise_levels["sp"]:
    noisy = noises.add_sp_noise_rgb(lena_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    sp_results.append(["Lena", "sp", level, ent, comp])
    
# for level in noise_levels["sp"]:
#     noisy = noises.add_sp_noise_rgb(cat_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     sp_results.append(["Cat", "sp", level, ent, comp])
    
# for level in noise_levels["sp"]:
#     noisy = noises.add_sp_noise_rgb(coffee_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     sp_results.append(["Coffee", "sp", level, ent, comp])
    
for level in noise_levels["sp"]:
    noisy = noises.add_sp_noise_rgb(rocket_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    sp_results.append(["Rocket", "sp", level, ent, comp])
    
# for level in noise_levels["sp"]:
#     noisy = noises.add_sp_noise_rgb(uniform_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     sp_results.append(["Uniform", "sp", level, ent, comp])
    
for level in noise_levels["sp"]:
    noisy = noises.add_sp_noise_rgb(chess_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    sp_results.append(["Chess", "sp", level, ent, comp])
    
# %% RUÍDO POISSON ##

poisson_results = []

for level in noise_levels["poisson"]:
    noisy = noises.add_poisson_noise_rgb(lena_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    poisson_results.append(["Lena", "poisson", level, ent, comp])
    
# for level in noise_levels["poisson"]:
#     noisy = noises.add_poisson_noise_rgb(cat_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     poisson_results.append(["Cat", "poisson", level, ent, comp])
    
# for level in noise_levels["poisson"]:
#     noisy = noises.add_poisson_noise_rgb(coffee_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     poisson_results.append(["Coffee", "poisson", level, ent, comp])
    
for level in noise_levels["poisson"]:
    print(level)
    noisy = noises.add_poisson_noise_rgb(rocket_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    poisson_results.append(["Rocket", "poisson", level, ent, comp]) #PROBLEMA AQUI
    
# for level in noise_levels["poisson"]:
#     noisy = noises.add_poisson_noise_rgb(uniform_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     poisson_results.append(["Uniform", "poisson", level, ent, comp])
    
for level in noise_levels["poisson"]:
    noisy = noises.add_poisson_noise_rgb(chess_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    poisson_results.append(["Chess", "poisson", level, ent, comp])
    
# %% RUÍDO SPECKLE ##

speckle_results = []

for level in noise_levels["speckle"]:
    noisy = noises.add_speckle_noise_rgb(lena_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    speckle_results.append(["Lena", "speckle", level, ent, comp])
    
# for level in noise_levels["speckle"]:
#     noisy = noises.add_speckle_noise_rgb(cat_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     speckle_results.append(["Cat", "speckle", level, ent, comp])
    
# for level in noise_levels["speckle"]:
#     noisy = noises.add_speckle_noise_rgb(coffee_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     speckle_results.append(["Coffee", "speckle", level, ent, comp])
    
for level in noise_levels["speckle"]:
    noisy = noises.add_speckle_noise_rgb(rocket_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    speckle_results.append(["Rocket", "speckle", level, ent, comp])
    
# for level in noise_levels["speckle"]:
#     noisy = noises.add_speckle_noise_rgb(uniform_3d, factor=level)
#     probs = bp.bandt_pompe_method(noisy, dx, dy)
#     ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
#     comp = bp.complexity(probs, ent)
#     speckle_results.append(["Uniform", "speckle", level, ent, comp])
    
for level in noise_levels["speckle"]:
    noisy = noises.add_speckle_noise_rgb(chess_3d, factor=level)
    probs = bp.bandt_pompe_method(noisy, dx, dy)
    ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
    comp = bp.complexity(probs, ent)
    speckle_results.append(["Chess", "speckle", level, ent, comp])
    
# %% GAUSSIAN

system = "Gaussian" # Gaussian, SP, Poisson, Speckle  ##AJUSTAR NA MÃO

df_noises = pd.DataFrame(
    gaussian_results, #gaussian_results, sp_results ##AJUSTAR NA MÃO
    columns=["Imagem", "Tipo_Ruido", "Intensidade", "Entropia", "Complexidade"]
)

fig, ax_main = plt.subplots(figsize=(9, 6))
bp.plot_3d(D)

# Paleta de cores para distinguir cada imagem
colors = {
    "Lena": "yellow",
    #"Cat": "pink",
    #"Coffee": "gold",
    "Rocket": "magenta",
    #"Uniform": "gray",
    "Chess": "cyan",
}

# Loop para plotar cada imagem com cor e rótulo próprios
for img_name, color in colors.items():
    subset = df_noises[df_noises["Imagem"] == img_name]
    ax_main.plot(
        subset["Entropia"],
        subset["Complexidade"],
        marker='o',
        linestyle='--',
        label=img_name + " + " + system + " Noise",
        color=color
    )

ax_main.plot(ent_lena_3d, comp_lena_3d, marker='o', 
             linestyle='--', label='Lena Astronaut', color='blue')

ax_main.plot(ent_chess_3d, comp_chess_3d, marker='o', 
             linestyle='--', label='Chess', color='red')

ax_main.plot(ent_rocket_3d, comp_rocket_3d, marker='o', 
             linestyle='--', label='Rocket', color='orange')

#PLOTS DOS RUÍDOS MÁXIMOS

probs = bp.bandt_pompe_method(gauss_max, dx, dy) ##AJUSTAR NA MÃO
ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
comp = bp.complexity(probs, ent)
ax_main.plot(ent, comp, marker='o', 
             linestyle='--', label= system + ' Noise', color='green')

# Configurações do gráfico
ax_main.set_title("MvACECP", fontsize=14)
ax_main.set_xlabel("Entropy", fontsize=12)
ax_main.set_ylabel("Complexity", fontsize=12)
ax_main.legend(title="Label", loc="upper left", fontsize=9)
plt.tight_layout()
#plt.show()

axins = plt.axes([1.05, 0.27, 0.6, 0.6])
#axins = ax.inset_axes([1.2, 0.2, 0.8, 0.8])

x1, x2, y1, y2 = 0.98, 0.995, 0.02, 0.03 # Gaussian ##AJUSTAR NA MÃO
#x1, x2, y1, y2 = 0.35, 0.8, 0.3, 0.45 # Salt and Pepper  
#x1, x2, y1, y2 = 0.9, 1.0, 0.0, 0.12 # Poisson
#x1, x2, y1, y2 = 0.98, 1.0, 0.0, 0.06 # Speckle

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#axins.set_xticklabels('')
#axins.set_yticklabels('')
plt.ticklabel_format(style='sci', axis='x', scilimits=(x1, x2)) #x1,x2
plt.ticklabel_format(style='sci', axis='y', scilimits=(y1, y2)) #y1,y2
    
ax_main.indicate_inset_zoom(axins, edgecolor = '0.4')
    
folder='./limits/mv/'
M = 240

df_cont = pd.read_csv(folder +'continua-N'+str(M)+'.q1', skiprows=20, sep = '  ', engine = 'python') #7
df_cont.columns = ['HT', 'CJT']

df_troz = pd.read_csv(folder +'trozos-N'+str(M)+'.q1', skiprows=20, sep = '  ', engine = 'python')
df_troz.columns = ['HT', 'CJT']

# Loop para plotar cada imagem com cor e rótulo próprios
for img_name, color in colors.items():
    subset = df_noises[df_noises["Imagem"] == img_name]
    axins.plot(
        subset["Entropia"],
        subset["Complexidade"],
        marker='o',
        linestyle='--',
        label=img_name,
        color=color
    )

axins.plot(ent_lena_3d, comp_lena_3d, marker='o', 
             linestyle='--', label='Lena', color='blue')

axins.plot(ent_chess_3d, comp_chess_3d, marker='o', 
             linestyle='--', label='Chess', color='red')

axins.plot(ent_rocket_3d, comp_rocket_3d, marker='o', 
             linestyle='--', label='Rocket', color='orange')

#PLOTS DOS RUÍDOS MÁXIMOS
axins.plot(ent, comp, marker='o', 
             linestyle='--', label= system + ' Noise', color='green')

axins.plot(df_cont['HT'], df_cont['CJT'], color='black')
axins.plot(df_troz['HT'], df_troz['CJT'], color='black')

# # Configurações do zoom do gráfico
# axins.set_title("MvCECP - Ruído Gaussiano", fontsize=14)
# axins.set_xlabel("Entropia", fontsize=12)
# axins.set_ylabel("Complexidade", fontsize=12)
# axins.legend(title="Imagem", loc="upper right", fontsize=9)

loc = "C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/plots/Ruidos/"
plt.savefig(loc + system + "_noise.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.savefig(loc + system + "_noise.png", format="png", dpi=300, bbox_inches="tight")
plt.show()

#%% SALT AND PEPPER

system = "SP" # Gaussian, SP, Poisson, Speckle  ##AJUSTAR NA MÃO

df_noises = pd.DataFrame(
    sp_results, #gaussian_results, sp_results ##AJUSTAR NA MÃO
    columns=["Imagem", "Tipo_Ruido", "Intensidade", "Entropia", "Complexidade"]
)

fig, ax_main = plt.subplots(figsize=(9, 6))
bp.plot_3d(D)

# Paleta de cores para distinguir cada imagem
colors = {
    "Lena": "yellow",
    #"Cat": "pink",
    #"Coffee": "gold",
    "Rocket": "magenta",
    #"Uniform": "gray",
    "Chess": "cyan",
}

# Loop para plotar cada imagem com cor e rótulo próprios
for img_name, color in colors.items():
    subset = df_noises[df_noises["Imagem"] == img_name]
    ax_main.plot(
        subset["Entropia"],
        subset["Complexidade"],
        marker='o',
        linestyle='--',
        label=img_name + " + " + system + " Noise",
        color=color
    )

ax_main.plot(ent_lena_3d, comp_lena_3d, marker='o', 
             linestyle='--', label='Lena Astronaut', color='blue')

ax_main.plot(ent_chess_3d, comp_chess_3d, marker='o', 
             linestyle='--', label='Chess', color='red')

ax_main.plot(ent_rocket_3d, comp_rocket_3d, marker='o', 
             linestyle='--', label='Rocket', color='orange')

#PLOTS DOS RUÍDOS MÁXIMOS

probs = bp.bandt_pompe_method(sp_max, dx, dy) ##AJUSTAR NA MÃO
ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
comp = bp.complexity(probs, ent)
ax_main.plot(ent, comp, marker='o', 
             linestyle='--', label= system + ' Noise', color='green')

# Configurações do gráfico
ax_main.set_title("MvACECP", fontsize=14)
ax_main.set_xlabel("Entropy", fontsize=12)
ax_main.set_ylabel("Complexity", fontsize=12)
ax_main.legend(title="Label", loc="upper left", fontsize=9)
plt.tight_layout()
#plt.show()

axins = plt.axes([1.05, 0.27, 0.6, 0.6])
#axins = ax.inset_axes([1.2, 0.2, 0.8, 0.8])

#x1, x2, y1, y2 = 0.98, 0.995, 0.02, 0.03 # Gaussian ##AJUSTAR NA MÃO
x1, x2, y1, y2 = 0.35, 0.8, 0.3, 0.45 # Salt and Pepper  
#x1, x2, y1, y2 = 0.9, 1.0, 0.0, 0.12 # Poisson
#x1, x2, y1, y2 = 0.98, 1.0, 0.0, 0.06 # Speckle

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#axins.set_xticklabels('')
#axins.set_yticklabels('')
plt.ticklabel_format(style='sci', axis='x', scilimits=(x1, x2)) #x1,x2
plt.ticklabel_format(style='sci', axis='y', scilimits=(y1, y2)) #y1,y2
    
ax_main.indicate_inset_zoom(axins, edgecolor = '0.4')
    
folder='./limits/mv/'
M = 240

df_cont = pd.read_csv(folder +'continua-N'+str(M)+'.q1', skiprows=20, sep = '  ', engine = 'python') #7
df_cont.columns = ['HT', 'CJT']

df_troz = pd.read_csv(folder +'trozos-N'+str(M)+'.q1', skiprows=20, sep = '  ', engine = 'python')
df_troz.columns = ['HT', 'CJT']

# Loop para plotar cada imagem com cor e rótulo próprios
for img_name, color in colors.items():
    subset = df_noises[df_noises["Imagem"] == img_name]
    axins.plot(
        subset["Entropia"],
        subset["Complexidade"],
        marker='o',
        linestyle='--',
        label=img_name,
        color=color
    )

axins.plot(ent_lena_3d, comp_lena_3d, marker='o', 
             linestyle='--', label='Lena', color='blue')

axins.plot(ent_chess_3d, comp_chess_3d, marker='o', 
             linestyle='--', label='Chess', color='red')

axins.plot(ent_rocket_3d, comp_rocket_3d, marker='o', 
             linestyle='--', label='Rocket', color='orange')

#PLOTS DOS RUÍDOS MÁXIMOS
axins.plot(ent, comp, marker='o', 
             linestyle='--', label= system + ' Noise', color='green')

axins.plot(df_cont['HT'], df_cont['CJT'], color='black')
axins.plot(df_troz['HT'], df_troz['CJT'], color='black')

# # Configurações do zoom do gráfico
# axins.set_title("MvCECP - Ruído Gaussiano", fontsize=14)
# axins.set_xlabel("Entropia", fontsize=12)
# axins.set_ylabel("Complexidade", fontsize=12)
# axins.legend(title="Imagem", loc="upper right", fontsize=9)

loc = "C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/plots/Ruidos/"
plt.savefig(loc + system + "_noise.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.savefig(loc + system + "_noise.png", format="png", dpi=300, bbox_inches="tight")
plt.show()

#%% SPECKLE

system = "Poisson" # Gaussian, SP, Poisson, Speckle  ##AJUSTAR NA MÃO

df_noises = pd.DataFrame(
    poisson_results, #gaussian_results, sp_results ##AJUSTAR NA MÃO
    columns=["Imagem", "Tipo_Ruido", "Intensidade", "Entropia", "Complexidade"]
)

fig, ax_main = plt.subplots(figsize=(9, 6))
bp.plot_3d(D)

# Paleta de cores para distinguir cada imagem
colors = {
    "Lena": "yellow",
    #"Cat": "pink",
    #"Coffee": "gold",
    "Rocket": "magenta",
    #"Uniform": "gray",
    "Chess": "cyan",
}

# Loop para plotar cada imagem com cor e rótulo próprios
for img_name, color in colors.items():
    subset = df_noises[df_noises["Imagem"] == img_name]
    ax_main.plot(
        subset["Entropia"],
        subset["Complexidade"],
        marker='o',
        linestyle='--',
        label=img_name + " + " + system + " Noise",
        color=color
    )

ax_main.plot(ent_lena_3d, comp_lena_3d, marker='o', 
             linestyle='--', label='Lena Astronaut', color='blue')

ax_main.plot(ent_chess_3d, comp_chess_3d, marker='o', 
             linestyle='--', label='Chess', color='red')

ax_main.plot(ent_rocket_3d, comp_rocket_3d, marker='o', 
             linestyle='--', label='Rocket', color='orange')

#PLOTS DOS RUÍDOS MÁXIMOS

probs = bp.bandt_pompe_method(poisson_max, dx, dy) ##AJUSTAR NA MÃO
ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
comp = bp.complexity(probs, ent)
ax_main.plot(ent, comp, marker='o', 
             linestyle='--', label= system + ' Noise', color='green')

# Configurações do gráfico
ax_main.set_title("MvACECP", fontsize=14)
ax_main.set_xlabel("Entropy", fontsize=12)
ax_main.set_ylabel("Complexity", fontsize=12)
ax_main.legend(title="Label", loc="upper left", fontsize=9)
plt.tight_layout()
#plt.show()

axins = plt.axes([1.05, 0.27, 0.6, 0.6])
#axins = ax.inset_axes([1.2, 0.2, 0.8, 0.8])

#x1, x2, y1, y2 = 0.98, 0.995, 0.02, 0.03 # Gaussian ##AJUSTAR NA MÃO
#x1, x2, y1, y2 = 0.35, 0.8, 0.3, 0.45 # Salt and Pepper  
x1, x2, y1, y2 = 0.9, 1.0, 0.0, 0.12 # Poisson
#x1, x2, y1, y2 = 0.98, 1.0, 0.0, 0.06 # Speckle

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#axins.set_xticklabels('')
#axins.set_yticklabels('')
plt.ticklabel_format(style='sci', axis='x', scilimits=(x1, x2)) #x1,x2
plt.ticklabel_format(style='sci', axis='y', scilimits=(y1, y2)) #y1,y2
    
ax_main.indicate_inset_zoom(axins, edgecolor = '0.4')
    
folder='./limits/mv/'
M = 240

df_cont = pd.read_csv(folder +'continua-N'+str(M)+'.q1', skiprows=20, sep = '  ', engine = 'python') #7
df_cont.columns = ['HT', 'CJT']

df_troz = pd.read_csv(folder +'trozos-N'+str(M)+'.q1', skiprows=20, sep = '  ', engine = 'python')
df_troz.columns = ['HT', 'CJT']

# Loop para plotar cada imagem com cor e rótulo próprios
for img_name, color in colors.items():
    subset = df_noises[df_noises["Imagem"] == img_name]
    axins.plot(
        subset["Entropia"],
        subset["Complexidade"],
        marker='o',
        linestyle='--',
        label=img_name,
        color=color
    )

axins.plot(ent_lena_3d, comp_lena_3d, marker='o', 
             linestyle='--', label='Lena', color='blue')

axins.plot(ent_chess_3d, comp_chess_3d, marker='o', 
             linestyle='--', label='Chess', color='red')

axins.plot(ent_rocket_3d, comp_rocket_3d, marker='o', 
             linestyle='--', label='Rocket', color='orange')

#PLOTS DOS RUÍDOS MÁXIMOS
axins.plot(ent, comp, marker='o', 
             linestyle='--', label= system + ' Noise', color='green')

axins.plot(df_cont['HT'], df_cont['CJT'], color='black')
axins.plot(df_troz['HT'], df_troz['CJT'], color='black')

# # Configurações do zoom do gráfico
# axins.set_title("MvCECP - Ruído Gaussiano", fontsize=14)
# axins.set_xlabel("Entropia", fontsize=12)
# axins.set_ylabel("Complexidade", fontsize=12)
# axins.legend(title="Imagem", loc="upper right", fontsize=9)

loc = "C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/plots/Ruidos/"
plt.savefig(loc + system + "_noise.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.savefig(loc + system + "_noise.png", format="png", dpi=300, bbox_inches="tight")
plt.show()

#%% SPECKLE

system = "Speckle" # Gaussian, SP, Poisson, Speckle  ##AJUSTAR NA MÃO

df_noises = pd.DataFrame(
    speckle_results, #gaussian_results, sp_results ##AJUSTAR NA MÃO
    columns=["Imagem", "Tipo_Ruido", "Intensidade", "Entropia", "Complexidade"]
)

fig, ax_main = plt.subplots(figsize=(9, 6))
bp.plot_3d(D)

# Paleta de cores para distinguir cada imagem
colors = {
    "Lena": "yellow",
    #"Cat": "pink",
    #"Coffee": "gold",
    "Rocket": "magenta",
    #"Uniform": "gray",
    "Chess": "cyan",
}

# Loop para plotar cada imagem com cor e rótulo próprios
for img_name, color in colors.items():
    subset = df_noises[df_noises["Imagem"] == img_name]
    ax_main.plot(
        subset["Entropia"],
        subset["Complexidade"],
        marker='o',
        linestyle='--',
        label=img_name + " + " + system + " Noise",
        color=color
    )

ax_main.plot(ent_lena_3d, comp_lena_3d, marker='o', 
             linestyle='--', label='Lena Astronaut', color='blue')

ax_main.plot(ent_chess_3d, comp_chess_3d, marker='o', 
             linestyle='--', label='Chess', color='red')

ax_main.plot(ent_rocket_3d, comp_rocket_3d, marker='o', 
             linestyle='--', label='Rocket', color='orange')

#PLOTS DOS RUÍDOS MÁXIMOS

probs = bp.bandt_pompe_method(speckle_max, dx, dy) ##AJUSTAR NA MÃO
ent = bp.shannon_entropy(probs, normalized=True, sigma=D)
comp = bp.complexity(probs, ent)
ax_main.plot(ent, comp, marker='o', 
             linestyle='--', label= system + ' Noise', color='green')

# Configurações do gráfico
ax_main.set_title("MvACECP", fontsize=14)
ax_main.set_xlabel("Entropy", fontsize=12)
ax_main.set_ylabel("Complexity", fontsize=12)
ax_main.legend(title="Label", loc="upper left", fontsize=9)
plt.tight_layout()
#plt.show()

axins = plt.axes([1.05, 0.27, 0.6, 0.6])
#axins = ax.inset_axes([1.2, 0.2, 0.8, 0.8])

#x1, x2, y1, y2 = 0.98, 0.995, 0.02, 0.03 # Gaussian ##AJUSTAR NA MÃO
#x1, x2, y1, y2 = 0.35, 0.8, 0.3, 0.45 # Salt and Pepper  
#x1, x2, y1, y2 = 0.9, 1.0, 0.0, 0.12 # Poisson
x1, x2, y1, y2 = 0.98, 1.0, 0.0, 0.06 # Speckle

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#axins.set_xticklabels('')
#axins.set_yticklabels('')
plt.ticklabel_format(style='sci', axis='x', scilimits=(x1, x2)) #x1,x2
plt.ticklabel_format(style='sci', axis='y', scilimits=(y1, y2)) #y1,y2
    
ax_main.indicate_inset_zoom(axins, edgecolor = '0.4')
    
folder='./limits/mv/'
M = 240

df_cont = pd.read_csv(folder +'continua-N'+str(M)+'.q1', skiprows=20, sep = '  ', engine = 'python') #7
df_cont.columns = ['HT', 'CJT']

df_troz = pd.read_csv(folder +'trozos-N'+str(M)+'.q1', skiprows=20, sep = '  ', engine = 'python')
df_troz.columns = ['HT', 'CJT']

# Loop para plotar cada imagem com cor e rótulo próprios
for img_name, color in colors.items():
    subset = df_noises[df_noises["Imagem"] == img_name]
    axins.plot(
        subset["Entropia"],
        subset["Complexidade"],
        marker='o',
        linestyle='--',
        label=img_name,
        color=color
    )

axins.plot(ent_lena_3d, comp_lena_3d, marker='o', 
             linestyle='--', label='Lena', color='blue')

axins.plot(ent_chess_3d, comp_chess_3d, marker='o', 
             linestyle='--', label='Chess', color='red')

axins.plot(ent_rocket_3d, comp_rocket_3d, marker='o', 
             linestyle='--', label='Rocket', color='orange')

#PLOTS DOS RUÍDOS MÁXIMOS
axins.plot(ent, comp, marker='o', 
             linestyle='--', label= system + ' Noise', color='green')

axins.plot(df_cont['HT'], df_cont['CJT'], color='black')
axins.plot(df_troz['HT'], df_troz['CJT'], color='black')

# # Configurações do zoom do gráfico
# axins.set_title("MvCECP - Ruído Gaussiano", fontsize=14)
# axins.set_xlabel("Entropia", fontsize=12)
# axins.set_ylabel("Complexidade", fontsize=12)
# axins.legend(title="Imagem", loc="upper right", fontsize=9)

loc = "C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/plots/Ruidos/"
plt.savefig(loc + system + "_noise.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.savefig(loc + system + "_noise.png", format="png", dpi=300, bbox_inches="tight")
plt.show()

#%%

all_results = []
for lst in (gaussian_results, sp_results, speckle_results, poisson_results):
    if lst:  # só concatena listas não vazias
        all_results.extend(lst)

# Cria DataFrame com as colunas desejadas
df_all = pd.DataFrame(all_results, columns=["Imagem", "Tipo_Ruido", "Intensidade", "Entropia", "Complexidade"])

# Assegura tipos apropriados
df_all["Imagem"] = df_all["Imagem"].astype(str)
df_all["Tipo_Ruido"] = df_all["Tipo_Ruido"].astype(str)
df_all["Intensidade"] = pd.to_numeric(df_all["Intensidade"], errors="coerce")
df_all["Entropia"] = pd.to_numeric(df_all["Entropia"], errors="coerce")
df_all["Complexidade"] = pd.to_numeric(df_all["Complexidade"], errors="coerce")

# Cria diretório de resultados (se não existir)
results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(results_dir, exist_ok=True)

# Caminhos dos arquivos
csv_out = os.path.join(results_dir, "mvacecp_ruidos_results.csv")
xlsx_out = os.path.join(results_dir, "mvacecp_ruidos_results.xlsx")

# Caso você esteja regravando um arquivo existente com colunas indesejadas,
# removemos explicitamente colunas 'timestamp', 'dx', 'dy', 'D' se existirem.
def _clean_and_save(df, csv_path, xlsx_path):
    # remove colunas indesejadas se existirem
    for col in ("timestamp", "dx", "dy", "D"):
        if col in df.columns:
            df = df.drop(columns=[col])
    # garante a ordem das colunas (apenas as 5 desejadas)
    desired_cols = ["Imagem", "Tipo_Ruido", "Intensidade", "Entropia", "Complexidade"]
    # se houver alguma coluna a mais, mantenha-a apenas se não for indesejada
    cols_present = [c for c in desired_cols if c in df.columns]
    df_to_save = df[cols_present].copy()
    # salva CSV e Excel
    df_to_save.to_csv(csv_path, index=False)
    try:
        df_to_save.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"Aviso: falha ao salvar Excel: {e} — apenas CSV foi salvo.")
    return df_to_save

df_saved = _clean_and_save(df_all, csv_out, xlsx_out)

print(f"Resultados agregados salvos em:\n - {csv_out}\n - {xlsx_out} (se compatível)")
print(f"Linhas salvas: {len(df_saved)}")