"""
==============================================================================
Análise Comparativa: Abordagem Multicanal vs Monocromática
Caracterização de Ruídos por Métricas de Entropia e Complexidade
── Versão SEM ponto de referência (intensidade nula excluída dos centróides) ──
==============================================================================

USO:
    python analise_centroides_sem_ref.py

CONFIGURAÇÃO:
    Edite a variável DATA_DIR abaixo para apontar para a pasta onde estão
    os 8 arquivos .xlsx de resultados.

ARQUIVOS ESPERADOS NA PASTA:
    mv_chroma_noise.xlsx        unv_chroma_noise.xlsx
    mv_color_misalignment.xlsx  unv_color_misalignment.xlsx
    mv_color_moire.xlsx         unv_color_moire.xlsx
    mv_jpeg_artifacts.xlsx      unv_jpeg_artifacts.xlsx

NOTA METODOLÓGICA:
    Os pontos com Intensidade = 0 (imagem original sem ruído) são plotados
    nas figuras de trajetória como referência visual, mas são EXCLUÍDOS do
    cálculo dos centróides, elipses de confiança e de todas as métricas
    estatísticas (distâncias, silhouette, FDR). Isso garante que os
    centróides representem exclusivamente o comportamento sob ruído ativo.

SAÍDAS GERADAS:
    1. fig1_scatter_multicanal.png       — Scatter com centróides e elipses (RGB)
    2. fig2_scatter_monocromatico.png    — Scatter com centróides e elipses (Mono)
    3. fig3_scatter_por_imagem.png       — Scatter por imagem (2×3 subplots)
    4. fig4_barras_distancias.png        — Gráfico de barras das distâncias entre centróides
    5. fig5_scatter_intensidades.png     — Trajetória dos pontos por nível de intensidade

DEPENDÊNCIAS:
    pip install pandas openpyxl numpy scipy matplotlib scikit-learn
==============================================================================
"""

# ===========================================================================
# >>>  CONFIGURE O CAMINHO DOS DADOS AQUI  <<<
# ===========================================================================
DATA_DIR = "C:/Users/givan/OneDrive/Área de Trabalho/Resultados Mestrado/"   # Ex: "C:/Users/Joao/dados" ou "/home/joao/resultados"

# ---------------------------------------------------------------------------
# Controle de exclusão da intensidade nula
# Altere para False para incluir o ponto de referência nos centróides
# ---------------------------------------------------------------------------
EXCLUIR_INTENSIDADE_ZERO = True
# ===========================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats
from itertools import combinations

# ---------------------------------------------------------------------------
# Paleta e estilo global
# ---------------------------------------------------------------------------
BG_DARK   = "#FFFFFF"
BG_PANEL  = "#F8F9FA"
BG_PANEL2 = "#F0F2F5"
GRID_COL  = "#CCCCCC"
TEXT_MAIN = "#1A1A2E"
TEXT_DIM  = "#555566"

NOISE_COLORS = {
    "Chroma Noise":        "#C62828",
    "Color Misalignment":  "#1565C0",
    "Color Moiré":         "#2E7D32",
    "JPEG Artifacts":      "#E65100",
}
NOISE_LABELS = {
    "Chroma Noise":        "Ruído de Crominância",
    "Color Misalignment":  "Desalinhamento Cromático",
    "Color Moiré":         "Moiré Cromático",
    "JPEG Artifacts":      "Artefatos JPEG",
}
IMAGE_MARKERS = {
    "Lena Astronauta": "o",
    "Foguete":         "s",
    "Padrão Xadrez":   "^",
}
IMAGE_COLORS_DARK = {
    "Lena Astronauta": "#6A1B9A",
    "Foguete":         "#00838F",
    "Padrão Xadrez":   "#F57F17",
}

INTENSITY_ALPHA = [0.30, 0.55, 0.78, 1.00]

plt.rcParams.update({
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    BG_PANEL,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_MAIN,
    "axes.titlecolor":   TEXT_MAIN,
    "xtick.color":       TEXT_DIM,
    "ytick.color":       TEXT_DIM,
    "text.color":        TEXT_MAIN,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.6,
    "legend.facecolor":  BG_PANEL2,
    "legend.edgecolor":  GRID_COL,
    "legend.labelcolor": TEXT_MAIN,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def confidence_ellipse(x, y, ax, n_std=1.8, edgecolor="white", alpha=0.18,
                        facecolor=None, linestyle="--", linewidth=1.4, zorder=1):
    """Desenha elipse de confiança (covariância) ao redor de um cluster."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle,
        edgecolor=edgecolor, facecolor=facecolor if facecolor else edgecolor,
        alpha=alpha if facecolor else 0, linestyle=linestyle,
        linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(ell)
    # borda sem fill
    ell2 = Ellipse(
        xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle,
        edgecolor=edgecolor, facecolor="none",
        linestyle=linestyle, linewidth=linewidth, zorder=zorder + 1, alpha=0.75,
    )
    ax.add_patch(ell2)


def draw_centroid(ax, cx, cy, color, size=220, zorder=15):
    """Marca o centróide com estrela de destaque e halo."""
    ax.scatter(cx, cy, s=size * 2.5, color=color, marker="*", alpha=0.15,
               zorder=zorder - 1, linewidths=0)
    ax.scatter(cx, cy, s=size, color=color, marker="*", edgecolors="black",
               linewidths=0.8, zorder=zorder,
               path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def annotate_centroid(ax, cx, cy, label, color, fontsize=8.5):
    """Texto com sombra ao lado do centróide."""
    ax.annotate(
        label, xy=(cx, cy),
        xytext=(5, 5), textcoords="offset points",
        fontsize=fontsize, color=color, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )


def styled_ax(ax, title="", xlabel="Entropia Normalizada",
              ylabel="Complexidade Estatística"):
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=9)


def legend_noise(extra_handles=None):
    handles = [
        mpatches.Patch(color=NOISE_COLORS[n], label=NOISE_LABELS[n])
        for n in NOISE_COLORS
    ]
    if extra_handles:
        handles += extra_handles
    return handles


def legend_images():
    return [
        Line2D([0], [0], marker=IMAGE_MARKERS[img], color="w",
               markerfacecolor="#AAAAAA", markersize=8,
               label=img, linestyle="None")
        for img in IMAGE_MARKERS
    ]


def centroid_star_handle():
    return Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD700",
                  markersize=11, label="Centróide (I>0)", linestyle="None")

def ref_point_handle():
    return Line2D([0], [0], marker="D", color="w", markerfacecolor="#FFFFFF",
                  markeredgecolor="#888888", markersize=7,
                  label="Referência (I=0, excluída do centróide)", linestyle="None")


# ---------------------------------------------------------------------------
# Carregar dados
# ---------------------------------------------------------------------------
def load_data(data_dir):
    noise_keys = ["chroma_noise", "color_misalignment", "color_moire", "jpeg_artifacts"]
    frames_mv, frames_unv = [], []
    missing = []
    for key in noise_keys:
        for prefix, frames in [("mv", frames_mv), ("unv", frames_unv)]:
            path = os.path.join(data_dir, f"{prefix}_{key}.xlsx")
            if not os.path.exists(path):
                missing.append(path)
            else:
                df = pd.read_excel(path)
                df["Abordagem"] = "Multicanal" if prefix == "mv" else "Monocromatico"
                frames.append(df)
    if missing:
        print("\n[ERRO] Arquivos não encontrados:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    mv_full  = pd.concat(frames_mv,  ignore_index=True)
    unv_full = pd.concat(frames_unv, ignore_index=True)

    # Traduzir nomes das imagens para português
    img_map = {"Lena": "Lena Astronauta", "Rocket": "Foguete", "Checkerboard": "Padrão Xadrez"}
    mv_full["Imagem"]  = mv_full["Imagem"].map(img_map)
    unv_full["Imagem"] = unv_full["Imagem"].map(img_map)

    # Versões filtradas (sem intensidade=0) usadas nos centróides e métricas
    mv_filt  = mv_full[mv_full["Intensidade"]   != 0].reset_index(drop=True)
    unv_filt = unv_full[unv_full["Intensidade"] != 0].reset_index(drop=True)

    return mv_full, unv_full, mv_filt, unv_filt


# ===========================================================================
# FIG 1 e 2 — Scatter full com centróides, elipses, por tipo de ruído
# ===========================================================================
def plot_scatter_full(df_full, df_filt, title, outfile, noises, abordagem_label):
    """
    df_full : todos os pontos (incluindo I=0) — usados apenas para plot
    df_filt : sem I=0 — usados para centróides e elipses
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)

    for noise in noises:
        color   = NOISE_COLORS[noise]
        sub_all  = df_full[df_full["Tipo_Ruido"] == noise]
        sub_filt = df_filt[df_filt["Tipo_Ruido"] == noise]

        xe_filt = sub_filt["Entropia"].values
        yc_filt = sub_filt["Complexidade"].values

        # Elipse de confiança — apenas pontos com ruído ativo
        confidence_ellipse(xe_filt, yc_filt, ax, n_std=1.8,
                           edgecolor=color, facecolor=color, alpha=0.10)

        # Pontos I=0 — marcador diamante, contorno tracejado, sem fill
        sub_ref = sub_all[sub_all["Intensidade"] == 0]
        for img in IMAGE_MARKERS:
            s = sub_ref[sub_ref["Imagem"] == img]
            ax.scatter(s["Entropia"], s["Complexidade"],
                       color="none", marker="D", s=62,
                       edgecolors=color, linewidths=1.2,
                       alpha=0.55, zorder=7, linestyle="--")

        # Pontos com ruído ativo (I > 0)
        for img, marker in IMAGE_MARKERS.items():
            s = sub_filt[sub_filt["Imagem"] == img]
            ax.scatter(s["Entropia"], s["Complexidade"],
                       color=color, marker=marker, s=72,
                       edgecolors="white", linewidths=0.5, alpha=0.88, zorder=8)

        # Centróide calculado apenas sobre I > 0
        cx, cy = xe_filt.mean(), yc_filt.mean()
        draw_centroid(ax, cx, cy, color)
        annotate_centroid(ax, cx, cy, NOISE_LABELS[noise], color, fontsize=8)

    styled_ax(ax, title=title)

    handles = (legend_noise()
               + [Line2D([0], [0], linestyle="--", color="#AAAAAA",
                         alpha=0.7, label="Elipse 1.8σ (I>0)")]
               + legend_images()
               + [centroid_star_handle(), ref_point_handle()])
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              framealpha=0.35, ncol=2)

    ax.text(0.99, 0.01, abordagem_label, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=TEXT_DIM, style="italic")

    plt.tight_layout(pad=1.4)
    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Salvo: {outfile}")


# ===========================================================================
# FIG 3 — Scatter 2×3: uma coluna por imagem, uma linha por abordagem
# ===========================================================================
def plot_scatter_por_imagem(mv_full, unv_full, mv_filt, unv_filt, noises, outfile):
    images = ["Lena Astronauta", "Foguete", "Padrão Xadrez"]
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG_DARK)
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30,
                  top=0.90, bottom=0.10, left=0.07, right=0.97)

    row_labels = ["Multicanal (RGB)", "Monocromático"]
    # (full, filt) por abordagem
    row_data = [(mv_full, mv_filt), (unv_full, unv_filt)]

    def get_limits(img):
        all_e, all_c = [], []
        for df_ab in [mv_full, unv_full]:
            sub = df_ab[df_ab["Imagem"] == img]
            all_e += sub["Entropia"].tolist()
            all_c += sub["Complexidade"].tolist()
        pad_e = (max(all_e) - min(all_e)) * 0.12
        pad_c = (max(all_c) - min(all_c)) * 0.12
        return (min(all_e) - pad_e, max(all_e) + pad_e,
                min(all_c) - pad_c, max(all_c) + pad_c)

    for row_i, ((df_full_ab, df_filt_ab), row_lbl) in enumerate(zip(row_data, row_labels)):
        for col_i, img in enumerate(images):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_facecolor(BG_PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_COL)

            xlim = get_limits(img)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(xlim[2], xlim[3])

            sub_full = df_full_ab[df_full_ab["Imagem"] == img]
            sub_filt = df_filt_ab[df_filt_ab["Imagem"] == img]

            for noise in noises:
                color = NOISE_COLORS[noise]

                sf = sub_filt[sub_filt["Tipo_Ruido"] == noise]
                xe, yc = sf["Entropia"].values, sf["Complexidade"].values

                # elipse sobre pontos filtrados
                if len(xe) >= 3:
                    confidence_ellipse(xe, yc, ax, n_std=1.5,
                                       edgecolor=color, facecolor=color,
                                       alpha=0.13, linestyle="--")

                # pontos I=0 como diamante vazado
                s0 = sub_full[(sub_full["Tipo_Ruido"] == noise) &
                               (sub_full["Intensidade"] == 0)]
                ax.scatter(s0["Entropia"], s0["Complexidade"],
                           color="none", marker="D", s=55,
                           edgecolors=color, linewidths=1.1,
                           alpha=0.50, zorder=7)

                # pontos I>0
                ax.scatter(xe, yc, color=color, marker="o", s=65,
                           edgecolors="white", linewidths=0.5,
                           alpha=0.9, zorder=8)

                # centróide sobre I>0
                if len(xe) > 0:
                    cx, cy = xe.mean(), yc.mean()
                    draw_centroid(ax, cx, cy, color, size=180)

            if row_i == 0:
                ax.set_title(img, fontsize=12, fontweight="bold",
                             color=TEXT_MAIN, pad=7)
            if col_i == 0:
                ax.set_ylabel(f"{row_lbl}\n\nComplexidade Estatística",
                              fontsize=9.5, color=TEXT_MAIN)
            else:
                ax.set_ylabel("")
            if row_i == 1:
                ax.set_xlabel("Entropia Normalizada", fontsize=9.5)
            else:
                ax.set_xlabel("")

            ax.tick_params(labelsize=8.5, colors=TEXT_DIM)
            ax.grid(True, alpha=0.3)

    handles = (legend_noise()
               + [centroid_star_handle(), ref_point_handle(),
                  Line2D([0], [0], linestyle="--", color="#AAAAAA",
                         alpha=0.7, label="Elipse 1.5σ (I>0)")])
    fig.legend(handles=handles, loc="lower center", ncol=7,
               fontsize=9, framealpha=0.25,
               bbox_to_anchor=(0.5, 0.02), fancybox=True)

    fig.suptitle("Espaço de Características por Imagem — Multicanal vs Monocromático\n"
                 "(centróides e elipses calculados sem o ponto de referência I=0)",
                 color=TEXT_MAIN, fontsize=14, fontweight="bold", y=0.97)

    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Salvo: {outfile}")


# ===========================================================================
# FIG 4 — Gráfico de barras das distâncias entre centróides
# ===========================================================================
def plot_barras_distancias(mv_filt, unv_filt, noises, outfile):
    """Distâncias calculadas exclusivamente com pontos I > 0."""
    pairs = list(combinations(noises, 2))

    mv_dists, unv_dists = [], []
    for n1, n2 in pairs:
        d_mv = np.linalg.norm(
            mv_filt[mv_filt["Tipo_Ruido"] == n1][["Entropia", "Complexidade"]].mean().values -
            mv_filt[mv_filt["Tipo_Ruido"] == n2][["Entropia", "Complexidade"]].mean().values)
        d_unv = np.linalg.norm(
            unv_filt[unv_filt["Tipo_Ruido"] == n1][["Entropia", "Complexidade"]].mean().values -
            unv_filt[unv_filt["Tipo_Ruido"] == n2][["Entropia", "Complexidade"]].mean().values)
        mv_dists.append(d_mv)
        unv_dists.append(d_unv)

    pair_labels = [
        f"{NOISE_LABELS[n1].split()[0]}\nvs\n{NOISE_LABELS[n2].split()[0]}"
        for n1, n2 in pairs
    ]
    # Labels mais curtos e legíveis
    short = {
        "Chroma Noise": "Crominância",
        "Color Misalignment": "Desalinhamento",
        "Color Moiré": "Moiré",
        "JPEG Artifacts": "JPEG",
    }
    pair_labels = [f"{short[n1]}\nvs {short[n2]}" for n1, n2 in pairs]

    x = np.arange(len(pairs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    bars1 = ax.bar(x - w / 2, mv_dists,  w, label="Multicanal (RGB)",
                   color="#1565C0", alpha=0.85, zorder=4)
    bars2 = ax.bar(x + w / 2, unv_dists, w, label="Monocromático",
                   color="#C62828", alpha=0.85, zorder=4)

    # Anotação de ganho relativo sobre a barra Multicanal
    for i, (mv_d, unv_d) in enumerate(zip(mv_dists, unv_dists)):
        gain = (mv_d - unv_d) / unv_d * 100 if unv_d > 0 else 0
        sign = "+" if gain >= 0 else ""
        color_ann = "#1565C0" if gain >= 0 else "#C62828"
        ax.annotate(
            f"{sign}{gain:.0f}%",
            xy=(x[i] - w / 2, mv_d),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
            color=color_ann, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    # Linhas de média
    ax.axhline(np.mean(mv_dists),  color="#1565C0", linestyle="--",
               linewidth=1.4, alpha=0.65,
               label=f"Média Multicanal ({np.mean(mv_dists):.4f})")
    ax.axhline(np.mean(unv_dists), color="#C62828", linestyle="--",
               linewidth=1.4, alpha=0.65,
               label=f"Média Monocromático ({np.mean(unv_dists):.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=9.5, color=TEXT_MAIN)
    ax.set_title("Distância Euclidiana entre Centróides dos Tipos de Ruído\n"
                 "no Espaço de Características (Entropia, Complexidade) — sem I=0",
                 fontweight="bold", pad=12)
    ax.set_ylabel("Distância Euclidiana", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.3, loc="upper left")
    ax.grid(True, axis="y", alpha=0.35)
    ax.tick_params(colors=TEXT_DIM)

    # Ganho médio global — posicionado no canto superior direito
    gain_global = (np.mean(mv_dists) - np.mean(unv_dists)) / np.mean(unv_dists) * 100
    ax.text(0.99, 0.97,
            f"Ganho médio Multicanal: +{gain_global:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#1565C0", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    plt.tight_layout(pad=1.4)
    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Salvo: {outfile}")


# ===========================================================================
# FIG 5 — Trajetória por nível de intensidade (setas de progressão)
# ===========================================================================
def plot_trajetoria_intensidades(mv_full, unv_full, mv_filt, unv_filt, noises, outfile):
    """
    Plota todos os pontos incluindo I=0 (como referência visual com marcador
    distinto ◇). As setas de progressão partem de I=0, mas o centróide de
    cada imagem/ruído é calculado apenas sobre I>0.
    Layout: 2 linhas (Multicanal / Mono) × 4 colunas (tipos de ruído).
    """
    images = ["Lena Astronauta", "Foguete", "Padrão Xadrez"]
    row_data   = [(mv_full, mv_filt), (unv_full, unv_filt)]
    row_labels = ["Multicanal (RGB)", "Monocromático"]

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor(BG_DARK)
    gs = GridSpec(2, 4, figure=fig, hspace=0.38, wspace=0.28,
                  top=0.88, bottom=0.12, left=0.06, right=0.98)

    for row_i, ((df_full_ab, df_filt_ab), row_lbl) in enumerate(zip(row_data, row_labels)):
        for col_i, noise in enumerate(noises):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_facecolor(BG_PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_COL)

            sub_full  = df_full_ab[df_full_ab["Tipo_Ruido"] == noise]
            sub_filt  = df_filt_ab[df_filt_ab["Tipo_Ruido"] == noise]
            noise_color = NOISE_COLORS[noise]

            for img in images:
                img_color = IMAGE_COLORS_DARK[img]
                marker    = IMAGE_MARKERS[img]

                # Todos os pontos ordenados por intensidade (inclui I=0)
                sub_all_img = sub_full[sub_full["Imagem"] == img].sort_values("Intensidade")
                xe_all = sub_all_img["Entropia"].values
                yc_all = sub_all_img["Complexidade"].values
                int_all = sub_all_img["Intensidade"].values

                # Plot de cada ponto
                for k, (e_val, c_val, intv) in enumerate(zip(xe_all, yc_all, int_all)):
                    if intv == 0:
                        # Referência: diamante vazado
                        ax.scatter(e_val, c_val, color="none", marker="D",
                                   s=72, edgecolors=img_color, linewidths=1.3,
                                   alpha=0.60, zorder=8)
                    else:
                        alpha = INTENSITY_ALPHA[min(k, len(INTENSITY_ALPHA) - 1)]
                        ax.scatter(e_val, c_val, color=img_color, marker=marker,
                                   s=80, alpha=alpha, edgecolors="white",
                                   linewidths=0.5, zorder=8)

                # Setas entre todos os pontos consecutivos (inclui I=0 → I1)
                for k in range(len(xe_all) - 1):
                    ax.annotate(
                        "", xy=(xe_all[k + 1], yc_all[k + 1]),
                        xytext=(xe_all[k], yc_all[k]),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=img_color, alpha=0.60,
                            lw=1.3, mutation_scale=10,
                        ),
                        zorder=7,
                    )

                # Centróide calculado APENAS sobre I>0
                sf_img = sub_filt[sub_filt["Imagem"] == img]
                if len(sf_img) > 0:
                    cx = sf_img["Entropia"].mean()
                    cy = sf_img["Complexidade"].mean()
                    draw_centroid(ax, cx, cy, img_color, size=150)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8, colors=TEXT_DIM)

            if row_i == 0:
                ax.set_title(NOISE_LABELS[noise], fontsize=10.5,
                             fontweight="bold", pad=7, color=noise_color)
            if col_i == 0:
                ax.set_ylabel(f"{row_lbl}\n\nComplexidade Estatística", fontsize=9.5)
            else:
                ax.set_ylabel("")
            if row_i == 1:
                ax.set_xlabel("Entropia", fontsize=9.5)
            else:
                ax.set_xlabel("")

    # Legenda global
    img_handles = [
        Line2D([0], [0], marker=IMAGE_MARKERS[img], color="w",
               markerfacecolor=IMAGE_COLORS_DARK[img],
               markersize=9, label=img, linestyle="None")
        for img in images
    ]
    fig.legend(
        handles=img_handles + [centroid_star_handle(), ref_point_handle()],
        loc="lower center", ncol=7, fontsize=9,
        framealpha=0.25, bbox_to_anchor=(0.5, 0.02),
    )

    fig.suptitle(
        "Trajetória no Espaço de Características por Nível de Intensidade\n"
        "(setas: progressão crescente · ◇ = referência I=0 · ★ = centróide calculado sem I=0)",
        color=TEXT_MAIN, fontsize=13, fontweight="bold", y=0.97,
    )

    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Salvo: {outfile}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("  Análise Comparativa Multicanal vs Monocromática")
    print("  (centróides calculados SEM ponto de referência I=0)")
    print("=" * 60)
    print(f"\n  Diretório de dados: {os.path.abspath(DATA_DIR)}\n")

    mv_full, unv_full, mv_filt, unv_filt = load_data(DATA_DIR)
    noises = list(mv_full["Tipo_Ruido"].unique())

    print(f"  Imagens   : {mv_full['Imagem'].unique().tolist()}")
    print(f"  Ruídos    : {noises}")
    print(f"  Registros (total)    : {len(mv_full)} MV / {len(unv_full)} UNV")
    print(f"  Registros (filtrado) : {len(mv_filt)} MV / {len(unv_filt)} UNV  (I>0)\n")
    print("  Gerando figuras...\n")

    out = DATA_DIR

    # --- FIG 1: Scatter Multicanal ---
    plot_scatter_full(
        mv_full, mv_filt, noises=noises,
        title="Espaço de Características — Abordagem Multicanal (RGB)\n"
              "(centróides e elipses calculados sem I=0)",
        abordagem_label="Imagens RGB · Análise Multicanal",
        outfile=os.path.join(out, "plots/fig1_scatter_multicanal.pdf"),
    )

    # --- FIG 2: Scatter Monocromático ---
    plot_scatter_full(
        unv_full, unv_filt, noises=noises,
        title="Espaço de Características — Abordagem Monocromática\n"
              "(centróides e elipses calculados sem I=0)",
        abordagem_label="Imagens em Grayscale · Análise Monocromática",
        outfile=os.path.join(out, "plots/fig2_scatter_monocromatico.pdf"),
    )

    # --- FIG 3: Scatter por imagem ---
    plot_scatter_por_imagem(
        mv_full, unv_full, mv_filt, unv_filt, noises=noises,
        outfile=os.path.join(out, "plots/fig3_scatter_por_imagem.pdf"),
    )

    # --- FIG 4: Barras distâncias ---
    plot_barras_distancias(
        mv_filt, unv_filt, noises=noises,
        outfile=os.path.join(out, "plots/fig4_barras_distancias.pdf"),
    )

    # --- FIG 5: Trajetória por intensidade ---
    plot_trajetoria_intensidades(
        mv_full, unv_full, mv_filt, unv_filt, noises=noises,
        outfile=os.path.join(out, "plots/fig5_scatter_intensidades.pdf"),
    )

    print("\n  Concluído! Todas as figuras foram salvas em:")
    print(f"  {os.path.abspath(out)}\n")


if __name__ == "__main__":
    main()
