from skimage import data
import matplotlib.pyplot as plt
import numpy as np

# Dimensoes das imagens
width, height = 512, 512

################### Carregar a imagem da Lena Astronauta ###################
lena_image = data.astronaut()/255.0
# lena_image = cv2.cvtColor(lena_image, cv2.COLOR_BGR2RGB)

# Plotar a imagem original
plt.imshow(lena_image)
plt.axis("off")
#plt.title("Lena Astronauta")
plt.show()

################### Carregar a imagem do Gato ###################

# Carregar a imagem do gato
cat_image = data.cat()/255.0

# Exibir a imagem
plt.imshow(cat_image)
plt.axis("off")  # Remove os eixos para melhor visualização
#plt.title("Gato")
plt.show()

################### Carregar a imagem do Café ###################

# Carregar a imagem do gato
coffee_image = data.coffee()/255.0

# Exibir a imagem
plt.imshow(coffee_image)
plt.axis("off")  # Remove os eixos para melhor visualização
#plt.title("Café")
plt.show()

################### Carregar a imagem do Foguete ###################

# Carregar a imagem do gato
rocket_image = data.rocket()/255.0

# Exibir a imagem
plt.imshow(rocket_image)
plt.axis("off")  # Remove os eixos para melhor visualização
#plt.title("Foguete")
plt.show()

################### Carregar a imagem Uniforme ###################

uniform_image = np.full((height, width, 3), 127, dtype=np.uint8)/255.0

# Exibir a imagem
plt.imshow(uniform_image)
plt.axis('off')
#plt.title("Uniforme")
plt.show()

################### Carregar a imagem do Xadrez Colorido ###################

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
            
chessboard_rgb = chessboard_rgb/255.0

# Exibir a imagem
plt.imshow(chessboard_rgb)
plt.axis('off')
#plt.title('Padrão Xadrez (Azul e Amarelo - RGB)')
plt.show()

# ################### Carregar a imagem de Ruídos Coloridos ###################

# # Criar uma matriz aleatória com valores 0, 1, 2 e 3 para representar as quatro cores
# random_pattern = np.random.choice([0, 1, 2, 3], size=(height, width), p=[1/4, 1/4, 1/4, 1/4])
    
# # Criar a imagem RGB baseada no padrão
# color_noise_image = np.zeros((height, width, 3), dtype=np.uint8)
# color_noise_image[random_pattern == 0] = [255, 255, 0]  # Amarelo
# color_noise_image[random_pattern == 1] = [0, 0, 255]  # Azul
# color_noise_image[random_pattern == 2] = [0, 255, 0]  # Verde
# color_noise_image[random_pattern == 2] = [255, 0, 0]  # Vermelho

# color_noise_image = color_noise_image/255.0

# # Exibir a imagem
# plt.imshow(color_noise_image)
# plt.axis("off")
# #plt.title("Ruído Colorido")
# plt.show()