import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import warnings
import math

# ---------------------------------------------------------------------
# Ajuste estes caminhos conforme seu ambiente
BASE_PATH = r"C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/KTH-TIPS2-b/"
LIMITS_FOLDER = r"./limits/mv/"   # pasta contendo continua-N{M}.q1 e trozos-N{M}.q1
# ---------------------------------------------------------------------

# Import suas libs (assumo que existem e estão no PYTHONPATH)
import lib_bp_img_3d_ord as bp

# Parâmetros de embedding
dx, dy = 2, 2
D = dx * dy    # ordem local usada no seu código original

# escolha sigma para a análise multivariada (3,4,5 ou 6)
sigma_mv = 4   # altere aqui para 3,4,5,6 conforme quiser
# mapa M correspondente (conforme seu código)
M_map = {3: 18, 4: 240, 5: 4200, 6: 90720}
M_for_sigma = M_map.get(sigma_mv, None)
if M_for_sigma is None:
    raise ValueError(f"sigma_mv={sigma_mv} inválido. Escolha 3,4,5 ou 6.")

# -------------------- Carregamento dos caminhos -----------------------
def gather_images(base_path):
    """
    Retorna:
      data: lista de arrays (H,W,3) RGB
      labels: np.array de labels das classes (material)
      file_paths: lista de caminhos completos para cada imagem (strings)
    """
    base_path = os.fspath(base_path)
    data = []
    labels = []
    file_paths = []
    # percorre materiais de forma ordenada para reprodutibilidade
    for material in sorted(os.listdir(base_path)):
        material_path = os.path.join(base_path, material)
        if not os.path.isdir(material_path):
            continue
        for sample in sorted(os.listdir(material_path)):
            sample_path = os.path.join(material_path, sample)
            if not os.path.isdir(sample_path):
                continue
            for img_file in sorted(os.listdir(sample_path)):
                # aceitar diversas extensões (minúsculas)
                if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    img_path = os.path.join(sample_path, img_file)
                    try:
                        # garante RGB (remove alfa se houver)
                        image = Image.open(img_path).convert("RGB")
                        image_array = np.array(image)  # formato (H,W,3) uint8, RGB
                        data.append(image_array)
                        labels.append(material)
                        file_paths.append(img_path)
                    except Exception as e:
                        print(f"Erro ao carregar {img_path}: {e}")
    return data, np.array(labels), file_paths

# Carrega imagens
data, labels, file_paths = gather_images(BASE_PATH)
print(f"Total imagens carregadas: {len(data)}")
unique_labels = np.unique(labels)
num_classes = len(unique_labels)
print(f"Classes detectadas ({num_classes}): {unique_labels}")

#%% PLOT DE 1 IMAGEM POR CATEGORIA #

# garante que labels é array numpy
labels = np.array(labels)

# encontra o índice da primeira imagem de cada classe
first_indices = []
for lbl in unique_labels:
    idxs = np.where(labels == lbl)[0]
    if len(idxs) > 0:
        first_indices.append(idxs[0])

# configura layout da grade (ajusta colunas conforme necessário)
cols = 4
rows = math.ceil(num_classes / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
axes = axes.flatten()

for ax in axes:
    ax.axis('off')

for i, idx in enumerate(first_indices):
    ax = axes[i]
    img = data[idx]
    # caso a imagem não seja RGB ou seja grayscale (HxW), trate com cmap
    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)  # espera RGB (PIL -> np.array produz RGB)
    ax.set_title(f"{unique_labels[i]}", fontsize=10)
    ax.axis('off')

# remover eixos vazios extras
for j in range(len(first_indices), len(axes)):
    axes[j].axis('off')

plt.suptitle("Primeira imagem de cada categoria (KTH-TIPS2-b)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%% -------------------- Cálculo MvCECP (3 canais) -----------------------
ent_color = []
comp_color = []

# Observação importante:
# - Estou assumindo que bp.bandt_pompe_method aceita um array RGB (H,W,3) uint8.
# - Se a sua função espera BGR (cv2), converta por: img_bgr = img_rgb[..., ::-1] antes de chamar.
#   --> ex: probs = bp.bandt_pompe_method(img_rgb[..., ::-1], dx, dy)

for i, img_color in enumerate(data, start=1):
    print(f"Processando imagem {i}/{len(data)}")
    try:
        # se sua lib espera BGR: img_in = img_color[..., ::-1]
        img_in = img_color
        probs = bp.bandt_pompe_method(img_in, dx, dy)
        # se probs vier como contagens, normalize:
        try:
            s = np.sum(probs)
            if s != 0 and not np.isclose(s, 1.0):
                probs = probs / s
        except Exception:
            pass
        h_color = bp.shannon_entropy(probs, normalized=True, sigma=D)
        c_color = bp.complexity(probs, h_color)
    except Exception as e:
        warnings.warn(f"Erro ao computar Mv para imagem {i}: {e}")
        h_color, c_color = np.nan, np.nan
    ent_color.append(h_color)
    comp_color.append(c_color)

ent_color = np.array(ent_color)
comp_color = np.array(comp_color)

#%% -------------------- Cria e salva planilha com resultados -----------------------

# Cria pasta results dentro do BASE_PATH (se não existir)
#results_dir = os.path.join(BASE_PATH, "results")
results_dir = "C:/Users/givan/OneDrive/Área de Trabalho/Github/[Versão Final] PPGI_UFAL/results/"
os.makedirs(results_dir, exist_ok=True)

# Monta DataFrame com colunas solicitadas:
# "Tipo de Material", "Caminho_Relativo", "Nome_Arquivo", "Entropia", "Complexidade"
rel_paths = [os.path.relpath(p, BASE_PATH) for p in file_paths]
file_names = [os.path.basename(p) for p in file_paths]

df = pd.DataFrame({
    "Tipo de Material": labels,
    "Caminho_Relativo": rel_paths,
    "Nome_Arquivo": file_names,
    "Entropia": ent_color,
    "Complexidade": comp_color
})

# Ordena opcionalmente por material e caminho
df = df.sort_values(["Tipo de Material", "Caminho_Relativo"]).reset_index(drop=True)

# Salva em CSV e Excel
csv_out = os.path.join(results_dir, "kth_tips2b_mv_cecp_results.csv")
xlsx_out = os.path.join(results_dir, "kth_tips2b_mv_cecp_results.xlsx")

df.to_csv(csv_out, index=False)
try:
    df.to_excel(xlsx_out, index=False)
except Exception as e:
    warnings.warn(f"Falha ao salvar Excel: {e} — apenas CSV foi salvo.")

print(f"Planilha salva em:\n - {csv_out}\n - {xlsx_out} (se compatível)")

# -------------------- (Opcional) Exibe resumo estatístico por material -----------------------
summary = df.groupby("Tipo de Material")[["Entropia", "Complexidade"]].agg(['mean','std','count'])
print("\nResumo por material:")
print(summary)

#%% PLOT
    
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