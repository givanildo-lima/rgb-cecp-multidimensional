import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import lib_bp_img_3d_ord as bp

# --- Configurações Iniciais ---
dx = 2
dy = 2
D = dx * dy

# Lista dos tipos de ruído que você está analisando
noises = ["chroma_noise", "color_misalignment", "color_moire", "jpeg_artifacts"]

# Dicionário para traduzir os nomes dos ruídos para exibição nos gráficos
noise_labels = {
    "chroma_noise": "Ruído de Crominância",
    "color_misalignment": "Desalinhamento Cromático",
    "color_moire": "Moiré Cromático",
    "jpeg_artifacts": "Artefatos de JPEG"
}

# Cores para diferenciar os ruídos nos gráficos
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# --- DEFINA A PASTA ONDE SEUS ARQUIVOS ESTÃO AQUI ---
base_directory = r"C:\Users\givan\OneDrive\Área de Trabalho\Resultados Mestrado"

# --- Mapeamento de nomes para títulos em português ---
abordagem_labels = {"MV": "Multidimensional", "UNV": "Unidimensional"}
image_name_labels = {
    "Lena": "Lena",
    "Rocket": "Foguete",
    "Checkerboard": "Padrão Xadrez",
    "Astronaut": "Astronauta"
}

# --- Função para Carregar e Preparar os Dados ---
def load_data(base_dir, prefix):
    all_dfs = []
    for noise in noises:
        filename = f"{prefix}_{noise}.xlsx"
        path = os.path.join(base_dir, filename)
        
        if os.path.exists(path):
            df = pd.read_excel(path)
            df.columns = [c.strip() for c in df.columns]
            df["Abordagem"] = prefix.upper()
            df["Ruído_PT"] = noise_labels.get(noise, noise)
            all_dfs.append(df)
        else:
            print(f"Aviso: Arquivo não encontrado em {path}")
    return pd.concat(all_dfs) if all_dfs else pd.DataFrame()

# --- Carregar os dados ---
df_mv = load_data(base_directory, "mv")
df_unv = load_data(base_directory, "unv")

if df_mv.empty or df_unv.empty:
    print("Erro: Não foi possível carregar os dados. Verifique o caminho da pasta e os nomes dos arquivos.")
    exit()

images = df_mv["Imagem"].unique()
os.makedirs("visualizacao_centroides", exist_ok=True)

# --- Loop Principal para Gerar Gráficos por Imagem e Abordagem ---
for img_name in images:
    for df_all, prefix_code in [(df_mv, "MV"), (df_unv, "UNV")]:
        # Criar a figura e o eixo com a nova dimensão
        fig, ax = plt.subplots(1, 1, figsize=(9, 6)) # ALTERADO AQUI: (9, 8) para (9, 6)
        plt.rcParams["font.family"] = "DejaVu Sans"
        
        # --- PLOTAR CURVAS LIMITE (FRONTEIRAS) ---
        # As funções bp.plot_x_d usam plt.gca() e plotam no eixo atual (ax).
        if prefix_code == "MV":
            bp.plot_3d(D) 
        else:
            bp.plot_2d(D) 

        df_img = df_all[df_all["Imagem"] == img_name]
        
        # Plotar pontos individuais de ruído (Intensidade > 0)
        df_noise_only = df_img[df_img["Intensidade"] > 0]
        sns.scatterplot(
            data=df_noise_only,
            x="Entropia",
            y="Complexidade",
            hue="Ruído_PT",
            style="Ruído_PT",
            s=80,
            alpha=0.4,
            ax=ax,
            legend=False,
            zorder=2
        )
        
        # Calcular e plotar centroides
        for idx, noise in enumerate(df_img["Ruído_PT"].unique()):
            df_noise = df_img[(df_img["Ruído_PT"] == noise) & (df_img["Intensidade"] > 0)]
            if not df_noise.empty:
                centroid_h = df_noise["Entropia"].mean()
                centroid_c = df_noise["Complexidade"].mean()
                
                # Plotar o centroide
                ax.scatter(
                    centroid_h, centroid_c, 
                    color=colors[idx % len(colors)], 
                    edgecolor="black", 
                    s=250, 
                    marker="o", 
                    label=f"Centroide: {noise}",
                    zorder=10
                )
                
                # Desenhar região de dispersão
                dists = np.sqrt((df_noise["Entropia"] - centroid_h)**2 + (df_noise["Complexidade"] - centroid_c)**2)
                radius = dists.max() if not dists.empty else 0
                circle = plt.Circle((centroid_h, centroid_c), radius, color=colors[idx % len(colors)], fill=True, alpha=0.2, zorder=1)
                ax.add_artist(circle)

        # Destacar o ponto original (I=0)
        orig = df_img[df_img["Intensidade"] == 0]
        if not orig.empty:
            ax.scatter(orig["Entropia"].iloc[0], orig["Complexidade"].iloc[0], color="black", marker="*", s=300, label=image_name_labels.get(img_name, img_name), zorder=15, edgecolor="white")

        # --- Definir os limites dos eixos para focar nos dados ---
        min_h, max_h = df_img["Entropia"].min(), df_img["Entropia"].max()
        min_c, max_c = df_img["Complexidade"].min(), df_img["Complexidade"].max()
        margin_h = (max_h - min_h) * 0.1
        margin_c = (max_c - min_c) * 0.1
        ax.set_xlim(min_h - margin_h, max_h + margin_h)
        ax.set_ylim(min_c - margin_c, max_c + margin_c)

        # Títulos e Rótulos
        abordagem_pt = abordagem_labels.get(prefix_code, prefix_code)
        image_pt = image_name_labels.get(img_name, img_name)
        ax.set_title(f"Abordagem {abordagem_pt} - Imagem: {image_pt}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Entropia (H)")
        ax.set_ylabel("Complexidade (C)")
        
        # Legenda ajustada
        ax.legend(loc="best", fontsize=8, frameon=True, markerscale=0.7, labelspacing=0.5) 

        plt.tight_layout()
        output_path = f"visualizacao_centroides/centroides_{img_name.lower()}_{prefix_code.lower()}_com_limites.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Gráfico com limites salvo para {img_name} ({prefix_code}) em: {output_path}")