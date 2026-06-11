import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import pandas as pd
import cv2

import lib_bp_img_3d as bp
import lib_bp_img as bp2
import gerar_imagens as dataset
import lib_ribeiro as ordpy

# Parametros para o calculo de entropia e complexidade
dx, dy = 2, 2 # Dimensoes de embedding
D = dx * dy

# Definir a semente global para reprodutibilidade
np.random.seed(42)

################################ IMAGENS GERAIS ###############################

######## LENA ASTROUNATA ########

print("LENA\n")
lena_3d = data.astronaut().astype(np.float32)/255.0

probs_lena_3d = bp.bandt_pompe_method(lena_3d, dx, dy)
ent_lena_3d = bp.shannon_entropy(probs_lena_3d, normalized=True, sigma = D)
comp_lena_3d = bp.complexity(probs_lena_3d, ent_lena_3d)

lena_2d = cv2.cvtColor(lena_3d, cv2.COLOR_BGR2GRAY)

h_lena_2d, c_lena_2d = ordpy.get_entropy_complexity(lena_2d, dx=dx, dy=dy, taux=1, tauy=1)

probs_lena_2d = bp2.bandt_pompe_method(lena_2d, dx, dy)
ent_lena_2d = bp2.shannon_entropy(probs_lena_2d, normalized=True, D=D)
comp_lena_2d = bp2.complexity(probs_lena_2d, ent_lena_2d)

print("ENT:", ent_lena_2d, ", COMP:", comp_lena_2d, "\n")
print("ENT:", h_lena_2d, ", COMP:", c_lena_2d, "\n")

#plt.imshow(lena_3d)
#plt.imshow(lena_2d, cmap='gray')

######## GATINHO ########

print("CAT\n")
cat_3d = data.cat().astype(np.float32)/255.0

probs_cat_3d = bp.bandt_pompe_method(cat_3d, dx, dy)
ent_cat_3d = bp.shannon_entropy(probs_cat_3d, normalized=True, sigma = D)
comp_cat_3d = bp.complexity(probs_cat_3d, ent_cat_3d)

cat_2d = cv2.cvtColor(cat_3d, cv2.COLOR_BGR2GRAY)

h_cat_2d, c_cat_2d = ordpy.get_entropy_complexity(cat_2d, dx=dx, dy=dy, taux=1, tauy=1)

probs_cat_2d = bp2.bandt_pompe_method(cat_2d, dx, dy)
ent_cat_2d = bp2.shannon_entropy(probs_cat_2d, normalized=True, D=D)
comp_cat_2d = bp2.complexity(probs_cat_2d, ent_cat_2d)

# plt.imshow(cat_3d)
# plt.imshow(cat_2d, cmap='gray')

######## XÍCARA DE CAFÉ ########

print("COFFEE\n")
coffee_3d = data.coffee().astype(np.float32)/255.0

probs_coffee_3d = bp.bandt_pompe_method(coffee_3d, dx, dy)
ent_coffee_3d = bp.shannon_entropy(probs_coffee_3d, normalized=True, sigma = D)
comp_coffee_3d = bp.complexity(probs_coffee_3d, ent_coffee_3d)

coffee_2d = cv2.cvtColor(coffee_3d, cv2.COLOR_BGR2GRAY)

h_coffee_2d, c_coffee_2d = ordpy.get_entropy_complexity(coffee_2d, dx=dx, dy=dy, taux=1, tauy=1)

probs_coffee_2d = bp2.bandt_pompe_method(coffee_2d, dx, dy)
ent_coffee_2d = bp2.shannon_entropy(probs_coffee_2d, normalized=True, D=D)
comp_coffee_2d = bp2.complexity(probs_coffee_2d, ent_coffee_2d)

# plt.imshow(coffee_3d)
# plt.imshow(coffee_2d, cmap='gray')

######## FOGUETE ########

print("ROCKET\n")
rocket_3d = data.rocket().astype(np.float32)/255.0

probs_rocket_3d = bp.bandt_pompe_method(rocket_3d, dx, dy)
ent_rocket_3d = bp.shannon_entropy(probs_rocket_3d, normalized=True, sigma = D)
comp_rocket_3d = bp.complexity(probs_rocket_3d, ent_rocket_3d)

rocket_2d = cv2.cvtColor(rocket_3d, cv2.COLOR_BGR2GRAY)

h_rocket_2d, c_rocket_2d = ordpy.get_entropy_complexity(rocket_2d, dx=dx, dy=dy, taux=1, tauy=1)

probs_rocket_2d = bp2.bandt_pompe_method(rocket_2d, dx, dy)
ent_rocket_2d = bp2.shannon_entropy(probs_rocket_2d, normalized=True, D=D)
comp_rocket_2d = bp2.complexity(probs_rocket_2d, ent_rocket_2d)

# plt.imshow(rocket_3d)
# plt.imshow(rocket_2d, cmap='gray')

######## IMAGEM UNIFORME ########

print("UNIFORM\n")
uniform_3d = dataset.uniform_image.astype(np.float32)

probs_uniform_3d = bp.bandt_pompe_method(uniform_3d, dx, dy)
ent_uniform_3d = bp.shannon_entropy(probs_uniform_3d, normalized=True, sigma = D)
comp_uniform_3d = bp.complexity(probs_uniform_3d, ent_uniform_3d)

uniform_2d = cv2.cvtColor(uniform_3d, cv2.COLOR_BGR2GRAY)

h_uniform_2d, c_uniform_2d = ordpy.get_entropy_complexity(uniform_2d, dx=dx, dy=dy, taux=1, tauy=1)

probs_uniform_2d = bp2.bandt_pompe_method(uniform_2d, dx, dy)
ent_uniform_2d = bp2.shannon_entropy(probs_uniform_2d, normalized=True, D=D)
comp_uniform_2d = bp2.complexity(probs_uniform_2d, ent_uniform_2d)

# plt.imshow(uniform_3d)
# plt.imshow(uniform_2d, cmap='gray')

######## XADREZ COLORIDO ########

print("CHESS\n")
chess_3d = dataset.chessboard_rgb.astype(np.float32)

probs_chess_3d = bp.bandt_pompe_method(chess_3d, dx, dy)
ent_chess_3d = bp.shannon_entropy(probs_chess_3d, normalized=True, sigma = D)
comp_chess_3d = bp.complexity(probs_chess_3d, ent_chess_3d)

chess_2d = cv2.cvtColor(chess_3d, cv2.COLOR_BGR2GRAY)

h_chess_2d, c_chess_2d = ordpy.get_entropy_complexity(chess_2d, dx=dx, dy=dy, taux=1, tauy=1)

probs_chess_2d = bp2.bandt_pompe_method(chess_2d, dx, dy)
ent_chess_2d = bp2.shannon_entropy(probs_chess_2d, normalized=True, D=D)
comp_chess_2d = bp2.complexity(probs_chess_2d, ent_chess_2d)

# plt.imshow(chess_3d)
# plt.imshow(chess_2d, cmap='gray')

######## RUÍDOS COLORIDOS ########

print("COLORED\n")
colored_3d = dataset.color_noise_image.astype(np.float32)

probs_colored_3d = bp.bandt_pompe_method(colored_3d, dx, dy)
ent_colored_3d = bp.shannon_entropy(probs_colored_3d, normalized=True, sigma = D)
comp_colored_3d = bp.complexity(probs_colored_3d, ent_colored_3d)

colored_2d = cv2.cvtColor(colored_3d, cv2.COLOR_BGR2GRAY)

h_colored_2d, c_colored_2d = ordpy.get_entropy_complexity(colored_2d, dx=dx, dy=dy, taux=1, tauy=1)

probs_colored_2d = bp2.bandt_pompe_method(colored_2d, dx, dy)
ent_colored_2d = bp2.shannon_entropy(probs_colored_2d, normalized=True, D=D)
comp_colored_2d = bp2.complexity(probs_colored_2d, ent_colored_2d)

# plt.imshow(colored_3d)
# plt.imshow(colored_2d, cmap='gray')

################## CÁLCULO DE ENTROPIA E COMPLEXIDADE 3D ##################

fig, ax_main = plt.subplots(figsize=(10, 6))
    
# Adicionar limites no grafico principal
bp.plot_3d(D)

# Plot no grafico principal
ax_main.plot(ent_lena_3d, comp_lena_3d, marker='o', label='Lena 3d', color='blue')
ax_main.plot(ent_cat_3d, comp_cat_3d, marker='o', label="Cat 3d", color='green')
ax_main.plot(ent_coffee_3d, comp_coffee_3d, marker='o', label="Coffee 3d", color='yellow')
ax_main.plot(ent_rocket_3d, comp_rocket_3d, marker='o', label="Rocket 3d", color='pink')
ax_main.plot(ent_uniform_3d, comp_uniform_3d, marker='o', label="Uniform 3d", color='gray')
ax_main.plot(ent_chess_3d, comp_chess_3d, marker='o', label="Chess 3d", color='red')
ax_main.plot(ent_colored_3d, comp_colored_3d, marker='o', label="Colored 3d", color='purple')
################################# 2d ####################################################
# ax_main.plot(ent_lena_2d, comp_lena_2d, marker='s', label='Lena 2d', color='blue')
# ax_main.plot(ent_cat_2d, comp_cat_2d, marker='s', label="Cat 2d", color='green')
# ax_main.plot(ent_coffee_2d, comp_coffee_2d, marker='s', label="Coffee 2d", color='yellow')
# ax_main.plot(ent_rocket_2d, comp_rocket_2d, marker='s', label="Rocket 2d", color='pink')
# ax_main.plot(ent_uniform_2d, comp_uniform_2d, marker='s', label="Uniform 2d", color='gray')
# ax_main.plot(ent_chess_2d, comp_chess_2d, marker='s', label="Chess 2d", color='red')
# ax_main.plot(ent_colored_2d, comp_colored_2d, marker='s', label="Colored 2d", color='purple')
# #########################################################################################

# Configuracoes do grafico principal
ax_main.set_title("MvCECP", fontsize=14)
ax_main.set_xlabel("Entropia", fontsize=12)
ax_main.set_ylabel("Complexidade", fontsize=12)
ax_main.legend(loc="upper left", fontsize=10)

plt.show()

################## CÁLCULO DE ENTROPIA E COMPLEXIDADE 2D Ribeiro ##################

fig, ax_main = plt.subplots(figsize=(10, 6))
    
# Adicionar limites no grafico principal
bp.plot_2d(D)

# Plot no grafico principal
ax_main.plot(h_lena_2d, c_lena_2d, marker='s', label='Lena Astronaut', color='blue')
ax_main.plot(h_cat_2d, c_cat_2d, marker='s', label="Cat", color='green')
ax_main.plot(h_coffee_2d, c_coffee_2d, marker='s', label="Coffee", color='yellow')
ax_main.plot(h_rocket_2d, c_rocket_2d, marker='s', label="Rocket", color='pink')
ax_main.plot(h_uniform_2d, c_uniform_2d, marker='s', label="Uniform", color='gray')
ax_main.plot(h_chess_2d, c_chess_2d, marker='s', label="Chessboard", color='red')
ax_main.plot(h_colored_2d, c_colored_2d, marker='s', label="Colored", color='purple')

# Configuracoes do grafico principal
ax_main.set_title("Ribeiro", fontsize=14)
ax_main.set_xlabel("Entropia", fontsize=12)
ax_main.set_ylabel("Complexidade", fontsize=12)
ax_main.legend(loc="upper left", fontsize=10)

plt.show()

################## CÁLCULO DE ENTROPIA E COMPLEXIDADE 2D DISTÂNCIA ##################

fig, ax_main = plt.subplots(figsize=(10, 6))
    
# Adicionar limites no grafico principal
bp.plot_3d(D)

# Plot no grafico principal
ax_main.plot(ent_lena_2d, comp_lena_2d, marker='s', label='Lena 2d', color='blue')
ax_main.plot(ent_cat_2d, comp_cat_2d, marker='s', label="Cat 2d", color='green')
ax_main.plot(ent_coffee_2d, comp_coffee_2d, marker='s', label="Coffee 2d", color='yellow')
ax_main.plot(ent_rocket_2d, comp_rocket_2d, marker='s', label="Rocket 2d", color='pink')
ax_main.plot(ent_uniform_2d, comp_uniform_2d, marker='s', label="Uniform 2d", color='gray')
ax_main.plot(ent_chess_2d, comp_chess_2d, marker='s', label="Chess 2d", color='red')
ax_main.plot(ent_colored_2d, comp_colored_2d, marker='s', label="Colored 2d", color='purple')

# Configuracoes do grafico principal
ax_main.set_title("Ribeiro", fontsize=14)
ax_main.set_xlabel("Entropia", fontsize=12)
ax_main.set_ylabel("Complexidade", fontsize=12)
ax_main.legend(loc="upper left", fontsize=10)

plt.show()