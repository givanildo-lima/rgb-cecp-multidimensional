"""
==============================================================================
Geração dos arquivos .xlsx para os canais R, G e B individuais
==============================================================================

DESCRIÇÃO:
    Aplica os mesmos ruídos e intensidades dos experimentos multicanal,
    extrai cada canal (R, G, B) e calcula Entropia e Complexidade Estatística
    via ordpy.complexity_entropy com dx=2, dy=2 — mesmos parâmetros usados
    na abordagem multicanal (lib_bp_img_3d_ord), garantindo comparação justa.

    Gera 12 arquivos .xlsx (3 canais × 4 tipos de ruído), no mesmo formato
    dos arquivos mv_*.xlsx e unv_*.xlsx existentes.

USO:
    1. Coloque este script na mesma pasta que rgb_noises.py
    2. Edite DATA_DIR abaixo (pasta onde os xlsx serão salvos)
    3. Execute:  python gerar_xlsx_canais_rgb.py

SAÍDAS:
    r_chroma_noise.xlsx          g_chroma_noise.xlsx          b_chroma_noise.xlsx
    r_color_misalignment.xlsx    g_color_misalignment.xlsx    b_color_misalignment.xlsx
    r_color_moire.xlsx           g_color_moire.xlsx           b_color_moire.xlsx
    r_jpeg_artifacts.xlsx        g_jpeg_artifacts.xlsx        b_jpeg_artifacts.xlsx

DEPENDÊNCIAS:
    pip install ordpy scikit-image openpyxl numpy pandas
==============================================================================
"""

# ===========================================================================
# >>>  CONFIGURE O CAMINHO DE SAÍDA AQUI  <<<
# ===========================================================================
DATA_DIR = "C:/Users/givan/OneDrive/Área de Trabalho/Resultados Mestrado/"   # Deixe vazio para salvar na mesma pasta deste script
                # Ou informe o caminho completo, ex:
                # DATA_DIR = r"C:\Users\givan\dados"
# ===========================================================================

import os
import sys
import numpy as np
import pandas as pd
from skimage import data as skdata

# Resolve diretório de saída
_script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = DATA_DIR.strip() if DATA_DIR.strip() else _script_dir
sys.path.insert(0, _script_dir)

try:
    import ordpy
except ImportError:
    print("[ERRO] ordpy não encontrado. Instale com: pip install ordpy")
    sys.exit(1)

try:
    import rgb_noises as rn
except ImportError:
    print("[ERRO] rgb_noises.py não encontrado. Coloque-o na mesma pasta deste script.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Parâmetros — dx=dy=2, mesmos do multicanal (lib_bp_img_3d_ord)
# ---------------------------------------------------------------------------
DX, DY = 2, 2

def calc_entropy_complexity_channel(channel_2d):
    """
    Entropia e Complexidade de um canal 2D via ordpy.complexity_entropy.
    dx=2, dy=2 garante paridade com o método multicanal (D = dx*dy = 4).
    """
    h, c = ordpy.complexity_entropy(channel_2d, dx=DX, dy=DY)
    return h, c

# ---------------------------------------------------------------------------
# Imagens — idêntico ao Experimentos_RGB.py
# ---------------------------------------------------------------------------
def _make_checkerboard(height=512, width=512, block_size=32):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    color1 = [0,   0,   255]   # Azul
    color2 = [255, 255,   0]   # Amarelo
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            c = color1 if (i // block_size + j // block_size) % 2 == 0 else color2
            img[i:i+block_size, j:j+block_size] = c
    return img

IMAGES = {
    "Lena":         lambda: skdata.astronaut().astype(np.uint8),
    "Rocket":       lambda: skdata.rocket().astype(np.uint8),
    "Checkerboard": lambda: _make_checkerboard(),
}

# ---------------------------------------------------------------------------
# Experimentos — idêntico ao Experimentos_RGB.py
# ---------------------------------------------------------------------------
NOISE_EXPERIMENTS = {
    "Chroma Noise": {
        "levels": [0.0, 0.02, 0.05, 0.1],
        "func":   lambda img, l: rn.add_chroma_noise(img, noise_strength=l, correlation=0.3),
        "key":    "chroma_noise",
    },
    "Color Misalignment": {
        "levels": [0.0, 0.5, 1.0, 2.0],
        "func":   lambda img, l: rn.add_color_misalignment(img, shift_range=(0, l)),
        "key":    "color_misalignment",
    },
    "Color Moiré": {
        "levels": [0.0, 0.1, 0.2, 0.3],
        "func":   lambda img, l: rn.add_color_moire(img, amplitude_range=(l, l + 1e-6)),
        "key":    "color_moire",
    },
    "JPEG Artifacts": {
        "levels": [100, 70, 40, 20],   # qualidade; 100 = sem ruído (I=0)
        "func":   lambda img, l: rn.add_jpeg_color_artifacts(img, quality=int(l)),
        "key":    "jpeg_artifacts",
    },
}

# Canal → índice no array RGB
CHANNELS = {"r": 0, "g": 1, "b": 2}

# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------
def run():
    print("=" * 62)
    print("  Geração de xlsx — Canais R, G e B individuais")
    print(f"  Parâmetros: dx={DX}, dy={DY}  (ordpy.complexity_entropy)")
    print("=" * 62)
    print(f"\n  Diretório de saída: {DATA_DIR}\n")

    # {canal: {noise_key: [rows]}}
    results = {ch: {cfg["key"]: [] for cfg in NOISE_EXPERIMENTS.values()}
               for ch in CHANNELS}

    total = len(IMAGES) * len(NOISE_EXPERIMENTS) * 4
    done  = 0

    for img_name, img_loader in IMAGES.items():
        image = img_loader()
        print(f"  Imagem: {img_name}  {image.shape}")

        for noise_name, cfg in NOISE_EXPERIMENTS.items():
            for level in cfg["levels"]:

                is_reference = (level == 0.0) or \
                               (noise_name == "JPEG Artifacts" and level == 100)

                if is_reference:
                    img_proc  = image
                    intensity = 0.0
                else:
                    img_proc  = cfg["func"](image, level)
                    intensity = level

                for ch_name, ch_idx in CHANNELS.items():
                    channel = img_proc[:, :, ch_idx]
                    ent, comp = calc_entropy_complexity_channel(channel)
                    results[ch_name][cfg["key"]].append({
                        "Imagem":       img_name,
                        "Tipo_Ruido":   noise_name,
                        "Intensidade":  intensity,
                        "Entropia":     ent,
                        "Complexidade": comp,
                    })

                done += 1
                print(f"    [{done/total*100:5.1f}%]  {img_name:13s} | "
                      f"{noise_name:20s} | I={intensity}", flush=True)

    print("\n  Salvando arquivos...\n")
    for ch_name in CHANNELS:
        for cfg in NOISE_EXPERIMENTS.values():
            key   = cfg["key"]
            df    = pd.DataFrame(results[ch_name][key])
            fname = os.path.join(DATA_DIR, f"{ch_name}_{key}.xlsx")
            df.to_excel(fname, index=False, engine="openpyxl")
            print(f"  Salvo: {fname}")

    print(f"\n  Concluído! 12 arquivos gerados em:\n  {DATA_DIR}\n")


if __name__ == "__main__":
    run()
