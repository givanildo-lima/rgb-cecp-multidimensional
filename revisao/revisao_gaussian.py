#loop_teste e o mais novo

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import lib_bp_img_3d as bp
import pandas as pd

# Parametros para o calculo de entropia e complexidade
dx, dy = 3, 2 # Dimensoes de embedding
D = dx * dy

# Definir a semente global para reprodutibilidade
np.random.seed(42)

# Dimensoes das imagens
width, height = 512, 512

############################ Lena Astronauta #################################

# Carregar a imagem Lena
lena_image = data.astronaut()
# lena_image = cv2.cvtColor(lena_image, cv2.COLOR_BGR2RGB)

# Plotar a imagem original
plt.imshow(lena_image)
plt.axis("off")
#plt.title("Imagem Original (Lena)")
plt.show()

# Calcular as probabilidades dos padroes ordinais
probs_lena = bp.bandt_pompe_method(lena_image, dx, dy)
    
# Calcular a entropia de Shannon normalizada
ent_lena = bp.shannon_entropy(probs_lena, normalized=True, sigma = D)
    
# Calcular a complexidade estatastica
comp_lena = bp.complexity(probs_lena, ent_lena)

######################## Imagem de Padrao Xadrez #############################

block_size = 32

# Criar uma matriz para a imagem
chessboard_rgb = np.zeros((height, width, 3), dtype=np.uint8)

# Definir cores para o xadrez (em RGB)
color1 = [0, 0, 255]  # Azul
color2 = [255, 255, 0]  # Amarelo

# Preencher a matriz com o padrao xadrez
for i in range(0, height, block_size):
    for j in range(0, width, block_size):
        if (i // block_size + j // block_size) % 2 == 0:
            chessboard_rgb[i:i+block_size, j:j+block_size] = color1
        else:
            chessboard_rgb[i:i+block_size, j:j+block_size] = color2

# Exibir a imagem
plt.imshow(chessboard_rgb)
plt.axis('off')
#plt.title('Padrão Xadrez (Azul e Amarelo - RGB)')
plt.show()

# Calcular as probabilidades dos padroes ordinais
probs_chess = bp.bandt_pompe_method(chessboard_rgb, dx, dy)
    
# Calcular a entropia de Shannon normalizada
ent_chess = bp.shannon_entropy(probs_chess, normalized=True, sigma = D)
    
# Calcular a complexidade estati­stica
comp_chess = bp.complexity(probs_chess, ent_chess)

############################ Plots Base ######################################

# fig, ax_main = plt.subplots(figsize=(10, 6))
    
# # Adicionar limites no grafico principal
# bp.plot_3d(D)

# # Plot no grafico principal
# ax_main.plot(ent_lena, comp_lena, marker='o', label='Lena Astronaut', color='blue')
# ax_main.plot(ent_chess, comp_chess, marker='o', label="Chessboard RGB", color='red')

# # Configuracoes do grafico principal
# ax_main.set_title("MvCECP", fontsize=14)
# ax_main.set_xlabel("Entropia", fontsize=12)
# ax_main.set_ylabel("Complexidade", fontsize=12)
# ax_main.legend(loc="upper left", fontsize=10)

# plt.show()    

##############################################################################

def add_gaussian_noise(imagem, media=0, sigma=25, same_noise=False): #rui­do aditivo
    if same_noise:
        # Gera um rui­do para cada pixel em apenas 1 canal (dimensao: altura x largura)
        ruido_2d = np.random.normal(media, sigma, imagem.shape[:2])
        # Repete o rui­do para os 3 canais, garantindo que cada pixel tera o mesmo valor em R, G e B
        ruido = np.stack([ruido_2d]*3, axis=-1)
    else:
        # Gera rui­do independente para cada pixel em cada canal (dimensao: altura x largura x 3)
        ruido = np.random.normal(media, sigma, imagem.shape)
    
    # Adiciona o rui­do a imagem
    imagem_ruidosa = imagem + ruido
    
    # Garante que os valores dos pixels permanecam no intervalo [0, 255]
    imagem_ruidosa = np.clip(imagem_ruidosa, 0, 255)
    
    return imagem_ruidosa.astype(np.uint8)

# Configuracao dos niveis de rui­do e tipos
noise_functions = {
    "Gaussiano": add_gaussian_noise
}

noise_levels = range(0, 21, 1)

# Aplicar os rui­dos e armazenar as imagens e metricas
results_lena = []
results_chess = []
results_noise = []
    
#############################################################################

blank_image = np.full((height, width, 3), 255, dtype=np.uint8)
probs_blank = bp.bandt_pompe_method(blank_image, dx, dy)
ent_blank = bp.shannon_entropy(probs_blank, normalized=True, sigma = D)
comp_blank = bp.complexity(probs_blank, ent_blank)    
    
noise_max = add_gaussian_noise(blank_image, media=0, sigma=70)

# # Exibir a imagem
# plt.imshow(blank_image)
# plt.imshow(noise_max)
# plt.axis('off')
# plt.title("Gaussian")
# plt.show()

# Calcular as probabilidades dos padroes ordinais
probs_noise = bp.bandt_pompe_method(noise_max, dx, dy)
        
# Calcular a entropia de Shannon normalizada
ent_noise = bp.shannon_entropy(probs_noise, normalized=True, sigma = D)
        
# Calcular a complexidade estati­stica
comp_noise = bp.complexity(probs_noise, ent_noise)
    
results_noise.append({"Noise Type": "Gaussian", "Entropy": ent_noise, "Complexity": comp_noise})

fig, ax_main = plt.subplots(figsize=(10, 6))
        
# Adicionar limites no grafico principal
bp.plot_3d(D)

# Plot no grafico principal
ax_main.plot(ent_lena, comp_lena, marker='o', label='Lena Astronaut', color='blue')
ax_main.plot(ent_chess, comp_chess, marker='o', label="Chessboard RGB", color='red')
ax_main.plot(ent_noise, comp_noise, marker='o', label="Gaussian", color='green')
ax_main.plot(ent_blank, comp_blank, marker='o', label="Blank Image", color='black')
    
# Configuracoes do grafico principal
ax_main.set_title("MvCECP", fontsize=14)
ax_main.set_xlabel("Entropia", fontsize=12)
ax_main.set_ylabel("Complexidade", fontsize=12)
ax_main.legend(loc="upper left", fontsize=10)

plt.show()    

##############################################################################
    
for level in noise_levels:
    print(level)
    noisy_image_lena = []
    noisy_image_lena = add_gaussian_noise(lena_image, media=0, sigma=level)
        
    # # Exibir a imagem com rui­do no ni­vel 50% como exemplo
    # if level == 50:
    #     plt.imshow(noisy_image_lena)
    #     plt.axis("off")
    #     plt.title(f"{noise_name} Noise (50%)")
    #     plt.show()
    
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(noisy_image_lena)
    plt.axis("off")
    plt.title(f"Gaussian Noise {level}")
    plt.savefig(f"C:/Users/givan/OneDrive/Área de Trabalho/Mestrado/Plots/Lena/PNG/Gaussian_Noise_{level}.png", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"C:/Users/givan/OneDrive/Área de Trabalho/Mestrado/Plots/Lena/PDF/Gaussian_Noise_{level}.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

    # Calcular entropia e complexidade
    probs = bp.bandt_pompe_method(noisy_image_lena, dx, dy)  
    entropy = bp.shannon_entropy(probs, normalized=True, sigma = D)
    comp = bp.complexity(probs, entropy)

    # Armazenar os resultados
    results_lena.append({"Noise Type": "Gaussian", "Noise Level": level, "Entropy": entropy, "Complexity": comp})
        
    ######################################################################
    
    noisy_image_chess = []
    noisy_image_chess = add_gaussian_noise(chessboard_rgb, media=0, sigma=level)
        
    # # Exibir a imagem com rui­do no ni­vel 50% como exemplo
    # if level == 50:
    #     plt.imshow(noisy_image_chess)
    #     plt.axis("off")
    #     plt.title(f"{noise_name} Noise (50%)")
    #     plt.show()
    
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(noisy_image_chess)
    plt.axis("off")
    plt.title(f"Gaussian Noise {level}")
    plt.savefig(f"C:/Users/givan/OneDrive/Área de Trabalho/Mestrado/Plots/Xadrez/PNG/Gaussian_Noise_{level}.png", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"C:/Users/givan/OneDrive/Área de Trabalho/Mestrado/Plots/Xadrez/PDF/Gaussian_Noise_{level}.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

    # Calcular entropia e complexidade
    probs = bp.bandt_pompe_method(noisy_image_chess, dx, dy)  
    entropy = bp.shannon_entropy(probs, normalized=True, sigma = D)
    comp = bp.complexity(probs, entropy)

    # Armazenar os resultados
    results_chess.append({"Noise Type": "Gaussian", "Noise Level": level, "Entropy": entropy, "Complexity": comp})
    
# ################################## PLOT ######################################

aux_ent_1 = [result["Entropy"] for result in results_lena]
aux_comp_1 = [result["Complexity"] for result in results_lena]

aux_ent_2 = [result["Entropy"] for result in results_chess]
aux_comp_2 = [result["Complexity"] for result in results_chess]

aux_ent_noise = [result["Entropy"] for result in results_noise]
aux_comp_noise = [result["Complexity"] for result in results_noise]

fig, ax_main = plt.subplots(figsize=(10, 6)) #figsize=(10, 6)
        
# Adicionar limites no grafico principal
bp.plot_3d(D)
    
# Plot no grafico principal
ax_main.plot(aux_ent_1, aux_comp_1, marker='o', linestyle='--', label="Gaussian-Lena", color='orange')
ax_main.plot(ent_lena, comp_lena, marker='o', label='Lena Astronauta', color='blue')
ax_main.plot(aux_ent_2, aux_comp_2, marker='o', linestyle='--', label="Gaussian-Xadrez", color='cyan')
ax_main.plot(ent_chess, comp_chess, marker='o', label="Xadrez RGB", color='red')
ax_main.plot(aux_ent_noise, aux_comp_noise, marker='o', label="Gaussian", color='green')
    
# Configuracoes do grafico principal
ax_main.set_title("MvCECP Gaussian", fontsize=14)
ax_main.set_xlabel("Entropia", fontsize=12)
ax_main.set_ylabel("Complexidade", fontsize=12)
ax_main.legend(loc="upper left", fontsize=10)
    
axins = plt.axes([1.05, 0.27, 0.6, 0.6])
#axins = ax.inset_axes([1.2, 0.2, 0.8, 0.8])
x1, x2, y1, y2 = 0.91, 1.0, 0.0, 0.11 #0.97, 0.99, 0.0, 0.05
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

# Plot no grafico principal
axins.plot(aux_ent_1, aux_comp_1, marker='o', linestyle='--', label="Gaussian-Lena", color='orange')
axins.plot(ent_lena, comp_lena, marker='o', label='Lena Astronauta', color='blue')
axins.plot(aux_ent_2, aux_comp_2, marker='o', linestyle='--', label="Gaussian-Xadrez", color='cyan')
axins.plot(ent_chess, comp_chess, marker='o', label="Xadrez Colorido", color='red')
axins.plot(aux_ent_noise, aux_comp_noise, marker='o', label="Gaussian", color='green')

axins.plot(df_cont['HT'], df_cont['CJT'], color='black')
axins.plot(df_troz['HT'], df_troz['CJT'], color='black')
    
plt.savefig("C:/Users/givan/OneDrive/Área de Trabalho/Mestrado/Plots/mv_gaussian.png", bbox_inches='tight', dpi = 300)
plt.savefig("C:/Users/givan/OneDrive/Área de Trabalho/Mestrado/Plots/mv_gaussian.pdf", bbox_inches='tight', dpi = 300)
    
plt.show()    