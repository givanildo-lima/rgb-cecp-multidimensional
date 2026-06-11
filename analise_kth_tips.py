import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Caminho raiz da base de dados
base_path = "C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/KTH-TIPS2-b/"

# Inicializa lista de dados
data = []
labels = []

# Percorre todas as pastas de materiais
for material in os.listdir(base_path):
    material_path = os.path.join(base_path, material)
    
    if os.path.isdir(material_path):
        # Percorre as amostras (sample_a, sample_b, ...)
        for sample in os.listdir(material_path):
            sample_path = os.path.join(material_path, sample)
            
            if os.path.isdir(sample_path):
                # Percorre todas as imagens dentro da amostra
                for img_file in os.listdir(sample_path):
                    if img_file.endswith(".png") or img_file.endswith(".jpg"):
                        img_path = os.path.join(sample_path, img_file)

                        try:
                            # Carrega a imagem (converte para numpy array)
                            image = Image.open(img_path)
                            #image = Image.open(img_path).convert('RGB')
                            #image = image.resize((200, 200))  # Padroniza tamanho se necessário
                            image_array = np.array(image)

                            # Armazena dados e rótulo
                            data.append(image_array)
                            labels.append(material)

                        except Exception as e:
                            print(f"Erro ao carregar {img_path}: {e}")
                            
############################### Ribeiro - 1 Canal ###############################
                            
import lib_ribeiro as ordpy
import cv2
import pandas as pd
import matplotlib
import math
from pathlib import Path

# Parametros para o calculo de entropia e complexidade
dx, dy = 2, 2 # Dimensoes de embedding
D = dx * dy

# Definir a semente global para reprodutibilidade
np.random.seed(42)

ent_gray = []
comp_gray = []
cont_gray = 0

for img in data:
    print("Cont Gray: ", cont_gray, "\n")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_gray, c_gray = ordpy.get_entropy_complexity(img_gray, dx=dx, dy=dy, taux=1, tauy=1)
    ent_gray.append(h_gray)
    comp_gray.append(c_gray)
    cont_gray += 1
    
def plot_2d(D = 6, folder='./limits/unv/'):

	fat = math.factorial(D)

	df_cont = pd.read_csv(folder +'continua-N'+str(fat)+'.q1', skiprows=20, sep = '  ', engine = 'python') #7
	df_cont.columns = ['HT', 'CJT']

	df_troz = pd.read_csv(folder +'trozos-N'+str(fat)+'.q1', skiprows=20, sep = '  ', engine = 'python')
	df_troz.columns = ['HT', 'CJT']

	plt.plot(df_cont['HT'], df_cont['CJT'], color='black')
	plt.plot(df_troz['HT'], df_troz['CJT'], color='black')

	plt.xlabel("Entropia de Permutação", size=15) #12
	plt.ylabel("Complexidade Estatística", size=15)

	return plt

# Converter listas para arrays NumPy
ent_gray = np.array(ent_gray)
comp_gray = np.array(comp_gray)
labels = np.array(labels)

# Obter os rótulos únicos
unique_labels = np.unique(labels)
num_classes = len(unique_labels)

# Preparação para o gráfico final
colors = matplotlib.colormaps['tab20']
color_list = [colors(i / (num_classes - 1)) for i in range(num_classes)]

fig, ax_main = plt.subplots(figsize=(10, 6))
plot_2d(D)

for i, label in enumerate(unique_labels):
    idx = labels == label
    ax_main.scatter(ent_gray[idx], comp_gray[idx],
                    color=color_list[i], label=label, alpha=0.7, s=20)

ax_main.set_title("Ribeiro", fontsize=14)
ax_main.set_xlabel("Entropia", fontsize=12)
ax_main.set_ylabel("Complexidade", fontsize=12)
ax_main.legend(loc="upper left", fontsize=10, title="Materiais")
plt.tight_layout()

axins = plt.axes([1.05, 0.27, 0.6, 0.6])
#axins = ax.inset_axes([1.2, 0.2, 0.8, 0.8])
x1, x2, y1, y2 = 0.95, 1.0, 0.0, 0.05
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#axins.set_xticklabels('')
#axins.set_yticklabels('')
plt.ticklabel_format(style='sci', axis='x', scilimits=(x1, x2)) #x1,x2
plt.ticklabel_format(style='sci', axis='y', scilimits=(y1, y2)) #y1,y2

ax_main.indicate_inset_zoom(axins, edgecolor = '0.4')

folder='./limits/unv/'
fat = np.math.factorial(D)

df_cont = pd.read_csv(folder +'continua-N'+str(fat)+'.q1', skiprows=20, sep = '  ') #7
df_cont.columns = ['HT', 'CJT']

df_troz = pd.read_csv(folder +'trozos-N'+str(fat)+'.q1', skiprows=20, sep = '  ')
df_troz.columns = ['HT', 'CJT']

# Plot no gráfico principal
for i, label in enumerate(unique_labels):
    idx = labels == label
    axins.scatter(ent_gray[idx], comp_gray[idx],
                    color=color_list[i], label=label, alpha=0.7, s=20)

axins.plot(df_cont['HT'], df_cont['CJT'], color='black')
axins.plot(df_troz['HT'], df_troz['CJT'], color='black')

parent_path = "C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/"

plt.savefig(parent_path + "/results/result_ribeiro.pdf", bbox_inches='tight')
plt.show()

############################### Giva - 3 Canais ###############################

import lib_bp_img_3d_ord as bp

ent_color = []
comp_color = []
cont_color = 1

for img_color in data:
    print("Cont Color: ", cont_color, "\n")
    probs = bp.bandt_pompe_method(img_color, dx, dy)
    h_color = bp.shannon_entropy(probs, normalized=True, sigma = D)
    c_color = bp.complexity(probs, h_color)
    ent_color.append(h_color)
    comp_color.append(c_color)
    cont_color += 1
    
def plot_3d(sigma = 3, folder='./limits/mv/'):
    if(sigma == 3):
        M = 18
    if(sigma == 4):
        M = 240
    if sigma == 5:
        M = 4200
    if sigma == 6:
        M = 90720

    df_cont = pd.read_csv(folder +'continua-N'+str(M)+'.q1', skiprows=7, sep = '  ', engine = 'python')
    df_cont.columns = ['HT', 'CJT']

    df_troz = pd.read_csv(folder +'trozos-N'+str(M)+'.q1', skiprows=7, sep = '  ', engine = 'python')
    df_troz.columns = ['HT', 'CJT']

    plt.plot(df_cont['HT'], df_cont['CJT'], color='black')
    plt.plot(df_troz['HT'], df_troz['CJT'], color='black')

    plt.xlabel("Entropia de Permutação", size=15) #size = 12
    plt.ylabel("Complexidade Estatística", size=15)
    #plt.title("Chaotic Attractors")

    return plt

# Converter listas para arrays NumPy
ent_color = np.array(ent_color)
comp_color = np.array(comp_color)
#labels = np.array(labels)

# Obter os rótulos únicos
unique_labels = np.unique(labels)
num_classes = len(unique_labels)

# Preparação para o gráfico final
colors = matplotlib.colormaps['tab10']
color_list = [colors(i / (num_classes - 1)) for i in range(num_classes)]

################################## PLOT #######################################

fig, ax_main = plt.subplots(figsize=(10, 6))
plot_3d(D)

for i, label in enumerate(unique_labels):
    idx = labels == label
    ax_main.scatter(ent_color[idx], comp_color[idx],
                    color=color_list[i], label=label, alpha=0.7, s=20)

ax_main.set_title("MvCECP", fontsize=14)
ax_main.set_xlabel("Entropia", fontsize=12)
ax_main.set_ylabel("Complexidade", fontsize=12)
ax_main.legend(loc="upper left", fontsize=10, title="Materiais")
plt.tight_layout()

axins = plt.axes([1.05, 0.27, 0.6, 0.6])
#axins = ax.inset_axes([1.2, 0.2, 0.8, 0.8])
x1, x2, y1, y2 = 0.9, 1.0, 0.0, 0.12
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#axins.set_xticklabels('')
#axins.set_yticklabels('')
plt.ticklabel_format(style='sci', axis='x', scilimits=(x1, x2)) #x1,x2
plt.ticklabel_format(style='sci', axis='y', scilimits=(y1, y2)) #y1,y2

ax_main.indicate_inset_zoom(axins, edgecolor = '0.4')

folder='./limits/mv/'
M = 240

df_cont = pd.read_csv(folder +'continua-N'+str(M)+'.q1', skiprows=20, sep = '  ') #7
df_cont.columns = ['HT', 'CJT']

df_troz = pd.read_csv(folder +'trozos-N'+str(M)+'.q1', skiprows=20, sep = '  ')
df_troz.columns = ['HT', 'CJT']

# Plot no gráfico principal
for i, label in enumerate(unique_labels):
    idx = labels == label
    axins.scatter(ent_color[idx], comp_color[idx],
                    color=color_list[i], label=label, alpha=0.7, s=20)

axins.plot(df_cont['HT'], df_cont['CJT'], color='black')
axins.plot(df_troz['HT'], df_troz['CJT'], color='black')

plt.show()