"""
==============================================================================
Análise Comparativa: Abordagem Multicanal vs Canais R, G e B
(sem Monocromático)
Caracterização de Ruídos por Métricas de Entropia e Complexidade Estatística
── Versão SEM ponto de referência (intensidade nula excluída dos centróides) ──
==============================================================================

USO:
    python analise_canais_rgb.py

CONFIGURAÇÃO:
    Edite a variável DATA_DIR abaixo para apontar para a pasta com os xlsx.

ARQUIVOS ESPERADOS NA PASTA:
    mv_*.xlsx   r_*.xlsx   g_*.xlsx   b_*.xlsx
    (4 arquivos por prefixo × 4 tipos de ruído = 16 arquivos no total)

NOTA SOBRE PARÂMETROS:
    Os canais individuais (R, G, B) são calculados com
    ordpy.complexity_entropy(dx=2, dy=2), garantindo paridade com o
    método multicanal (lib_bp_img_3d_ord, D = dx*dy = 4).

NOTA METODOLÓGICA:
    Pontos com Intensidade = 0 são plotados como referência visual (◇) mas
    EXCLUÍDOS do cálculo dos centróides, elipses e métricas estatísticas.

SAÍDAS GERADAS:
    fig1_scatter_comparativo.png    — Scatter 1×4: MV vs R vs G vs B (todos os dados)
    fig2_scatter_por_imagem.png     — Scatter 4×3: abordagem × imagem
    fig3_barras_distancias.png      — Distâncias entre centróides por abordagem
    fig4_trajetoria_intensidades.png — Trajetória por intensidade (4×4 subplots)

DEPENDÊNCIAS:
    pip install pandas openpyxl numpy scipy matplotlib scikit-learn
==============================================================================
"""

# ===========================================================================
# >>>  CONFIGURE O CAMINHO DOS DADOS AQUI  <<<
# ===========================================================================
DATA_DIR = "C:/Users/givan/OneDrive/Área de Trabalho/Resultados Mestrado/"   # Deixe vazio para usar automaticamente a pasta do script
                # Ou informe o caminho completo, ex:
                # DATA_DIR = r"C:\Users\givan\dados"
# ===========================================================================

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from itertools import combinations

# ---------------------------------------------------------------------------
# Paleta e estilo
# ---------------------------------------------------------------------------
BG_DARK   = "#FFFFFF"
BG_PANEL  = "#F8F9FA"
BG_PANEL2 = "#F0F2F5"
GRID_COL  = "#CCCCCC"
TEXT_MAIN = "#1A1A2E"
TEXT_DIM  = "#555566"

# Cor por abordagem
ABORDAGEM_COLORS = {
    "Multicanal": "#1565C0",   # azul escuro
    "Canal R":    "#C62828",   # vermelho
    "Canal G":    "#2E7D32",   # verde escuro
    "Canal B":    "#1A237E",   # azul índigo
}
ABORDAGEM_LABELS = {
    "Multicanal": "Multicanal (RGB)",
    "Canal R":    "Canal R",
    "Canal G":    "Canal G",
    "Canal B":    "Canal B",
}

NOISE_COLORS = {
    "Chroma Noise":        "#E53935",
    "Color Misalignment":  "#1E88E5",
    "Color Moiré":         "#43A047",
    "JPEG Artifacts":      "#FB8C00",
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
IMAGE_COLORS = {
    "Lena Astronauta": "#6A1B9A",
    "Foguete":         "#00838F",
    "Padrão Xadrez":   "#F57F17",
}

INTENSITY_ALPHA = [0.30, 0.55, 0.78, 1.00]

ABORDAGEM_ORDER = ["Multicanal", "Canal R", "Canal G", "Canal B"]

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
    "axes.titlesize":    12,
    "axes.labelsize":    10,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def confidence_ellipse(x, y, ax, n_std=1.8, edgecolor="gray", facecolor=None,
                        alpha=0.12, linestyle="--", linewidth=1.3, zorder=1):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.abs(vals))
    for fill, falpha in [(True, alpha), (False, 0)]:
        ell = Ellipse(
            xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle,
            edgecolor=edgecolor,
            facecolor=facecolor if (fill and facecolor) else "none",
            alpha=falpha if fill else 0.7,
            linestyle=linestyle, linewidth=linewidth,
            zorder=zorder + (0 if fill else 1),
        )
        ax.add_patch(ell)


def draw_centroid(ax, cx, cy, color, size=200, zorder=15):
    ax.scatter(cx, cy, s=size * 2.2, color=color, marker="*",
               alpha=0.15, zorder=zorder - 1, linewidths=0)
    ax.scatter(cx, cy, s=size, color=color, marker="*",
               edgecolors="black", linewidths=0.7, zorder=zorder,
               path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


def annotate_centroid(ax, cx, cy, label, color, fontsize=7.5):
    ax.annotate(label, xy=(cx, cy), xytext=(5, 5),
                textcoords="offset points", fontsize=fontsize,
                color=color, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


def style_ax(ax, title="", xlabel="Entropia Normalizada",
             ylabel="Complexidade Estatística"):
    ax.set_title(title, pad=8, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=8.5)


def centroid_handle():
    return Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD700",
                  markeredgecolor="black", markersize=11,
                  label="Centróide (I>0)", linestyle="None")

def ref_handle():
    return Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                  markeredgecolor="#888888", markersize=7,
                  label="Referência (I=0)", linestyle="None")

def noise_handles():
    return [mpatches.Patch(color=NOISE_COLORS[n], label=NOISE_LABELS[n])
            for n in NOISE_COLORS]

def abordagem_handles(subset=None):
    keys = subset if subset else ABORDAGEM_ORDER
    return [mpatches.Patch(color=ABORDAGEM_COLORS[k], label=ABORDAGEM_LABELS[k])
            for k in keys]

# ---------------------------------------------------------------------------
# Carregamento dos dados
# ---------------------------------------------------------------------------
NOISE_KEYS = ["chroma_noise", "color_misalignment", "color_moire", "jpeg_artifacts"]

PREFIX_MAP = {
    "Multicanal": "mv",
    "Canal R":    "r",
    "Canal G":    "g",
    "Canal B":    "b",
}

IMG_NAME_MAP = {
    "Lena":         "Lena Astronauta",
    "Rocket":       "Foguete",
    "Checkerboard": "Padrão Xadrez",
}


def load_data(data_dir):
    missing = []
    dfs = {}   # {abordagem: DataFrame completo}

    for abord, prefix in PREFIX_MAP.items():
        frames = []
        for key in NOISE_KEYS:
            path = os.path.join(data_dir, f"{prefix}_{key}.xlsx")
            if not os.path.exists(path):
                missing.append(path)
            else:
                df = pd.read_excel(path)
                df["Abordagem"] = abord
                frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined["Imagem"] = combined["Imagem"].map(
                lambda v: IMG_NAME_MAP.get(v, v))
            dfs[abord] = combined

    if missing:
        print("\n[ERRO] Arquivos não encontrados:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    # Versões filtradas (sem I=0)
    dfs_filt = {k: v[v["Intensidade"] != 0].reset_index(drop=True)
                for k, v in dfs.items()}
    return dfs, dfs_filt


# ===========================================================================
# FIG 1 — Scatter comparativo: 1 linha × 4 colunas (uma por abordagem)
#          Todos os tipos de ruído coloridos; I=0 como diamante
# ===========================================================================
def plot_scatter_comparativo(dfs_full, dfs_filt, noises, outfile):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6),
                              sharex=False, sharey=False)
    fig.patch.set_facecolor(BG_DARK)
    fig.subplots_adjust(wspace=0.30, top=0.88, bottom=0.13,
                        left=0.06, right=0.97)

    for col_i, abord in enumerate(ABORDAGEM_ORDER):
        ax = axes[col_i]
        ax.set_facecolor(BG_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)

        df_full = dfs_full[abord]
        df_filt = dfs_filt[abord]

        for noise in noises:
            color   = NOISE_COLORS[noise]
            sf      = df_filt[df_filt["Tipo_Ruido"] == noise]
            xe, yc  = sf["Entropia"].values, sf["Complexidade"].values

            confidence_ellipse(xe, yc, ax, n_std=1.8,
                               edgecolor=color, facecolor=color)

            # I=0 como diamante vazado
            s0 = df_full[(df_full["Tipo_Ruido"] == noise) &
                          (df_full["Intensidade"] == 0)]
            ax.scatter(s0["Entropia"], s0["Complexidade"],
                       color="none", marker="D", s=55,
                       edgecolors=color, linewidths=1.1,
                       alpha=0.50, zorder=7)

            # Pontos I>0 (marcador por imagem)
            for img, marker in IMAGE_MARKERS.items():
                s = sf[sf["Imagem"] == img]
                ax.scatter(s["Entropia"], s["Complexidade"],
                           color=color, marker=marker, s=65,
                           edgecolors="white", linewidths=0.4,
                           alpha=0.88, zorder=8)

            # Centróide
            if len(xe) > 0:
                cx, cy = xe.mean(), yc.mean()
                draw_centroid(ax, cx, cy, color)
                annotate_centroid(ax, cx, cy,
                                  NOISE_LABELS[noise].split()[0], color)

        style_ax(ax,
                 title=ABORDAGEM_LABELS[abord],
                 xlabel="Entropia Normalizada",
                 ylabel="Complexidade Estatística" if col_i == 0 else "")
        if col_i > 0:
            ax.set_ylabel("")

        # Faixa colorida no topo indicando abordagem
        abord_color = ABORDAGEM_COLORS[abord]
        ax.axhline(ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                   color=abord_color, linewidth=3, alpha=0.0)
        for sp in ["top", "bottom", "left", "right"]:
            ax.spines[sp].set_edgecolor(abord_color)
            ax.spines[sp].set_linewidth(1.8)

    # Legenda global
    handles = noise_handles() + [centroid_handle(), ref_handle(),
               Line2D([0], [0], linestyle="--", color="#AAAAAA",
                      alpha=0.7, label="Elipse 1.8σ")] + \
              [Line2D([0], [0], marker=IMAGE_MARKERS[img], color="w",
                      markerfacecolor="#888888", markersize=8,
                      label=img, linestyle="None")
               for img in IMAGE_MARKERS]
    fig.legend(handles=handles, loc="lower center", ncol=9,
               fontsize=8.5, framealpha=0.4,
               bbox_to_anchor=(0.5, 0.01), fancybox=True)

    fig.suptitle(
        "Espaço de Características — Comparação entre Abordagem Multicanal e Canais Individuais\n"
        "(centróides e elipses calculados sem I=0)",
        color=TEXT_MAIN, fontsize=13, fontweight="bold", y=0.97)

    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Salvo: {outfile}")


# ===========================================================================
# FIG 2 — Scatter 4×3: linhas = abordagem, colunas = imagem
# ===========================================================================
def plot_scatter_por_imagem(dfs_full, dfs_filt, noises, outfile):
    images = list(IMAGE_MARKERS.keys())

    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor(BG_DARK)
    gs = GridSpec(4, 3, figure=fig, hspace=0.38, wspace=0.28,
                  top=0.93, bottom=0.07, left=0.08, right=0.97)

    def get_limits(img):
        all_e, all_c = [], []
        for df_ab in dfs_full.values():
            sub = df_ab[df_ab["Imagem"] == img]
            all_e += sub["Entropia"].tolist()
            all_c += sub["Complexidade"].tolist()
        pe_ = (max(all_e) - min(all_e)) * 0.12
        pc_ = (max(all_c) - min(all_c)) * 0.12
        return (min(all_e) - pe_, max(all_e) + pe_,
                min(all_c) - pc_, max(all_c) + pc_)

    for row_i, abord in enumerate(ABORDAGEM_ORDER):
        for col_i, img in enumerate(images):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_facecolor(BG_PANEL)
            abord_color = ABORDAGEM_COLORS[abord]
            for sp in ax.spines.values():
                sp.set_edgecolor(abord_color)
                sp.set_linewidth(1.6)

            xlim = get_limits(img)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(xlim[2], xlim[3])

            df_full_ab = dfs_full[abord]
            df_filt_ab = dfs_filt[abord]

            sub_full = df_full_ab[df_full_ab["Imagem"] == img]
            sub_filt = df_filt_ab[df_filt_ab["Imagem"] == img]

            for noise in noises:
                color = NOISE_COLORS[noise]
                sf = sub_filt[sub_filt["Tipo_Ruido"] == noise]
                xe, yc = sf["Entropia"].values, sf["Complexidade"].values

                if len(xe) >= 3:
                    confidence_ellipse(xe, yc, ax, n_std=1.5,
                                       edgecolor=color, facecolor=color,
                                       alpha=0.12, linestyle="--")

                # I=0
                s0 = sub_full[(sub_full["Tipo_Ruido"] == noise) &
                               (sub_full["Intensidade"] == 0)]
                ax.scatter(s0["Entropia"], s0["Complexidade"],
                           color="none", marker="D", s=50,
                           edgecolors=color, linewidths=1.0,
                           alpha=0.50, zorder=7)

                ax.scatter(xe, yc, color=color, marker="o", s=60,
                           edgecolors="white", linewidths=0.4,
                           alpha=0.88, zorder=8)

                if len(xe) > 0:
                    draw_centroid(ax, xe.mean(), yc.mean(), color, size=160)

            # Títulos e rótulos
            if row_i == 0:
                ax.set_title(img, fontsize=11, fontweight="bold",
                             color=TEXT_MAIN, pad=7)
            if col_i == 0:
                ax.set_ylabel(
                    f"{ABORDAGEM_LABELS[abord]}\n\nComplexidade Estatística",
                    fontsize=9, color=abord_color, fontweight="bold")
            else:
                ax.set_ylabel("")
            if row_i == 3:
                ax.set_xlabel("Entropia Normalizada", fontsize=9)
            else:
                ax.set_xlabel("")

            ax.tick_params(labelsize=8, colors=TEXT_DIM)
            ax.grid(True, alpha=0.30)

    handles = (noise_handles()
               + [centroid_handle(), ref_handle(),
                  Line2D([0], [0], linestyle="--", color="#AAAAAA",
                         alpha=0.7, label="Elipse 1.5σ (I>0)")])
    fig.legend(handles=handles, loc="lower center", ncol=7,
               fontsize=9, framealpha=0.35,
               bbox_to_anchor=(0.5, 0.01), fancybox=True)

    fig.suptitle(
        "Espaço de Características por Imagem — Multicanal vs Canais R, G e B\n"
        "(centróides e elipses calculados sem I=0)",
        color=TEXT_MAIN, fontsize=13, fontweight="bold", y=0.975)

    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Salvo: {outfile}")


# ===========================================================================
# FIG 3 — Barras: distâncias entre centróides por abordagem
# ===========================================================================
def plot_barras_distancias(dfs_filt, noises, outfile):
    pairs = list(combinations(noises, 2))
    short = {
        "Chroma Noise":       "Crominância",
        "Color Misalignment": "Desalinhamento",
        "Color Moiré":        "Moiré",
        "JPEG Artifacts":     "JPEG",
    }
    pair_labels = [f"{short[n1]}\nvs {short[n2]}" for n1, n2 in pairs]

    dists = {}
    for abord in ABORDAGEM_ORDER:
        df = dfs_filt[abord]
        d_list = []
        for n1, n2 in pairs:
            c1 = df[df["Tipo_Ruido"] == n1][["Entropia","Complexidade"]].mean().values
            c2 = df[df["Tipo_Ruido"] == n2][["Entropia","Complexidade"]].mean().values
            d_list.append(np.linalg.norm(c1 - c2))
        dists[abord] = d_list

    x = np.arange(len(pairs))
    n_abord = len(ABORDAGEM_ORDER)
    total_w = 0.75
    w = total_w / n_abord

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)

    offsets = np.linspace(-(total_w - w) / 2, (total_w - w) / 2, n_abord)
    mv_dists = dists["Multicanal"]

    for i, (abord, offset) in enumerate(zip(ABORDAGEM_ORDER, offsets)):
        color = ABORDAGEM_COLORS[abord]
        d_list = dists[abord]
        ax.bar(x + offset, d_list, w,
               label=ABORDAGEM_LABELS[abord],
               color=color, alpha=0.85, zorder=4)

        # Ganho relativo ao Multicanal (apenas nos canais)
        if abord != "Multicanal":
            for j, (mv_d, ch_d) in enumerate(zip(mv_dists, d_list)):
                gain = (mv_d - ch_d) / ch_d * 100 if ch_d > 0 else 0
                sign = "+" if gain >= 0 else ""
                ann_color = ABORDAGEM_COLORS["Multicanal"] if gain >= 0 else color
                ax.annotate(
                    f"{sign}{gain:.0f}%",
                    xy=(x[j] + offset, max(mv_d, ch_d)),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5,
                    color=ann_color, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.8,
                                                foreground="white")],
                )

    # Linhas de média
    for abord in ABORDAGEM_ORDER:
        color = ABORDAGEM_COLORS[abord]
        mean_d = np.mean(dists[abord])
        ax.axhline(mean_d, color=color, linestyle="--",
                   linewidth=1.3, alpha=0.60,
                   label=f"Média {ABORDAGEM_LABELS[abord]} ({mean_d:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=9.5, color=TEXT_MAIN)
    ax.set_title(
        "Distância Euclidiana entre Centróides dos Tipos de Ruído\n"
        "no Espaço de Características (Entropia, Complexidade) — sem I=0",
        fontweight="bold", pad=12)
    ax.set_ylabel("Distância Euclidiana", fontsize=11)
    ax.tick_params(colors=TEXT_DIM)
    ax.grid(True, axis="y", alpha=0.35)

    # Legenda barras (superior esquerdo) separada das linhas de média
    bar_handles = [mpatches.Patch(color=ABORDAGEM_COLORS[a],
                                   label=ABORDAGEM_LABELS[a])
                   for a in ABORDAGEM_ORDER]
    ax.legend(handles=bar_handles, fontsize=9, framealpha=0.35,
              loc="upper left")

    # Ganho médio do Multicanal sobre cada canal (canto superior direito)
    mv_mean = np.mean(mv_dists)
    gain_lines = ["Ganho médio Multicanal:"]
    for abord in ["Canal R", "Canal G", "Canal B"]:
        ch_mean = np.mean(dists[abord])
        g = (mv_mean - ch_mean) / ch_mean * 100
        gain_lines.append(f"  vs {abord}: {'+' if g>=0 else ''}{g:.1f}%")
    ax.text(0.99, 0.97, "\n".join(gain_lines),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=ABORDAGEM_COLORS["Multicanal"],
            fontweight="bold", linespacing=1.5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    plt.tight_layout(pad=1.4)
    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Salvo: {outfile}")


# ===========================================================================
# FIG 4 — Trajetória por intensidade: 4 linhas (abordagem) × 4 colunas (ruído)
# ===========================================================================
def plot_trajetoria_intensidades(dfs_full, dfs_filt, noises, outfile):
    images = list(IMAGE_MARKERS.keys())

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(BG_DARK)
    gs = GridSpec(4, 4, figure=fig, hspace=0.38, wspace=0.28,
                  top=0.92, bottom=0.08, left=0.06, right=0.98)

    for row_i, abord in enumerate(ABORDAGEM_ORDER):
        for col_i, noise in enumerate(noises):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_facecolor(BG_PANEL)
            abord_color = ABORDAGEM_COLORS[abord]
            noise_color = NOISE_COLORS[noise]
            for sp in ax.spines.values():
                sp.set_edgecolor(abord_color)
                sp.set_linewidth(1.5)

            df_full_ab = dfs_full[abord]
            df_filt_ab = dfs_filt[abord]

            sub_full = df_full_ab[df_full_ab["Tipo_Ruido"] == noise]
            sub_filt = df_filt_ab[df_filt_ab["Tipo_Ruido"] == noise]

            for img in images:
                img_color  = IMAGE_COLORS[img]
                marker     = IMAGE_MARKERS[img]

                sub_all_img = sub_full[sub_full["Imagem"] == img].sort_values("Intensidade")
                xe_all = sub_all_img["Entropia"].values
                yc_all = sub_all_img["Complexidade"].values
                int_all = sub_all_img["Intensidade"].values

                for k, (e_val, c_val, intv) in enumerate(zip(xe_all, yc_all, int_all)):
                    if intv == 0:
                        ax.scatter(e_val, c_val, color="none", marker="D",
                                   s=65, edgecolors=img_color,
                                   linewidths=1.2, alpha=0.55, zorder=8)
                    else:
                        alpha = INTENSITY_ALPHA[min(k, len(INTENSITY_ALPHA)-1)]
                        ax.scatter(e_val, c_val, color=img_color,
                                   marker=marker, s=72, alpha=alpha,
                                   edgecolors="white", linewidths=0.4, zorder=8)

                # Setas
                for k in range(len(xe_all) - 1):
                    ax.annotate("",
                        xy=(xe_all[k+1], yc_all[k+1]),
                        xytext=(xe_all[k], yc_all[k]),
                        arrowprops=dict(arrowstyle="-|>", color=img_color,
                                        alpha=0.55, lw=1.2, mutation_scale=9),
                        zorder=7)

                # Centróide sobre I>0
                sf_img = sub_filt[sub_filt["Imagem"] == img]
                if len(sf_img) > 0:
                    draw_centroid(ax, sf_img["Entropia"].mean(),
                                  sf_img["Complexidade"].mean(),
                                  img_color, size=140)

            ax.grid(True, alpha=0.28)
            ax.tick_params(labelsize=7.5, colors=TEXT_DIM)

            if row_i == 0:
                ax.set_title(NOISE_LABELS[noise], fontsize=9.5,
                             fontweight="bold", pad=6, color=noise_color)
            if col_i == 0:
                ax.set_ylabel(
                    f"{ABORDAGEM_LABELS[abord]}\n\nComplexidade Estatística",
                    fontsize=8.5, color=abord_color, fontweight="bold")
            else:
                ax.set_ylabel("")
            if row_i == 3:
                ax.set_xlabel("Entropia Normalizada", fontsize=8.5)
            else:
                ax.set_xlabel("")

    # Legenda global
    img_handles = [
        Line2D([0], [0], marker=IMAGE_MARKERS[img], color="w",
               markerfacecolor=IMAGE_COLORS[img], markersize=9,
               label=img, linestyle="None")
        for img in images
    ]
    fig.legend(
        handles=img_handles + [centroid_handle(), ref_handle()],
        loc="lower center", ncol=6, fontsize=9,
        framealpha=0.30, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Trajetória no Espaço de Características por Nível de Intensidade\n"
        "Multicanal vs Canais R, G e B  ·  (◇ = referência I=0 · ★ = centróide sem I=0)",
        color=TEXT_MAIN, fontsize=13, fontweight="bold", y=0.975)

    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Salvo: {outfile}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    data_dir = DATA_DIR.strip() if DATA_DIR.strip() else \
               os.path.dirname(os.path.abspath(__file__))

    print("=" * 62)
    print("  Análise: Multicanal vs Canais R, G e B Individuais")
    print("  (centróides calculados SEM ponto de referência I=0)")
    print("=" * 62)
    print(f"\n  Diretório de dados: {os.path.abspath(data_dir)}\n")

    dfs_full, dfs_filt = load_data(data_dir)
    noises = list(dfs_full["Multicanal"]["Tipo_Ruido"].unique())

    print(f"  Abordagens : {list(dfs_full.keys())}")
    print(f"  Ruídos     : {noises}")
    for abord, df in dfs_filt.items():
        print(f"  {abord:12s}: {len(df)} registros (I>0)")
    print("\n  Gerando figuras...\n")

    out = data_dir

    plot_scatter_comparativo(
        dfs_full, dfs_filt, noises,
        outfile=os.path.join(out, "fig1_scatter_comparativo_mv_rgb.pdf"))

    plot_scatter_por_imagem(
        dfs_full, dfs_filt, noises,
        outfile=os.path.join(out, "fig2_scatter_por_imagem_mv_rgb.pdf"))

    plot_barras_distancias(
        dfs_filt, noises,
        outfile=os.path.join(out, "fig3_barras_distancias_mv_rgb.pdf"))

    plot_trajetoria_intensidades(
        dfs_full, dfs_filt, noises,
        outfile=os.path.join(out, "fig4_trajetoria_mv_rgb.pdf"))

    print(f"\n  Concluído! Figuras salvas em:\n  {os.path.abspath(out)}\n")


if __name__ == "__main__":
    main()
