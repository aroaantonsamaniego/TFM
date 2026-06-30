"""
1. Genera un PNG de visualización con los puntos coloreados sobre la imagen BN.
2. Guarda un TIFF multicanal (ImageJ/Fiji hyperstack) con 4 canales:
      Canal 1 → imagen original (blend de canales del TIFF de entrada)
      Canal 2 → máscara Borde    (discos gaussianos, valores 0-255)
      Canal 3 → máscara Aislada  (discos gaussianos, valores 0-255)
      Canal 4 → máscara Interior (discos gaussianos, valores 0-255)

   Fiji lo abre directamente como hyperstack multicanal (C=4, Z=1, T=1).
   Usa Make Composite + Channels Tool para asignar colores.

3. (Opcional) Calcula métricas de clasificación (TP, FP, TN, FN, Precision,
   Recall, F1, Specificity por clase + Accuracy y Kappa global) comparando
   un CSV con ground truth y otro con las predicciones. Guarda los resultados
   en un CSV.

Uso:
    python imagenes_clasificado.py

Dependencias:
    pip install pandas matplotlib pillow scipy tifffile
"""

import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.ndimage import gaussian_filter
import tifffile

COLOR_BORDE    = "#E53935"
COLOR_INTERIOR = "#1E88E5"
COLOR_AISLADA  = "#FF8C00"
COLOR_OTRO     = "#AAAAAA"


# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else arr.astype(np.float32)


def cargar_blend(ruta):
    """Carga TIFF (cualquier nº de canales/frames) y devuelve imagen BN float32 0-1."""
    img = Image.open(ruta)
    frames = []
    try:
        i = 0
        while True:
            img.seek(i)
            frames.append(norm(np.array(img).astype(np.float32)))
            i += 1
    except EOFError:
        pass
    return frames[0] if len(frames) == 1 else np.mean(frames, axis=0)


def asignar_color(c):
    c = str(c).strip().lower()
    if c == 'borde':    return COLOR_BORDE
    if c == 'interior': return COLOR_INTERIOR
    if c == 'aislada':  return COLOR_AISLADA
    return COLOR_OTRO


def mascara_gaussiana(h, w, puntos_x, puntos_y, sigma):
    """Crea una máscara uint8 con un disco gaussiano suave en cada punto."""
    canvas = np.zeros((h, w), dtype=np.float32)
    for x, y in zip(puntos_x, puntos_y):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w:
            canvas[yi, xi] = 1.0
    suave = gaussian_filter(canvas, sigma=sigma)
    if suave.max() > 0:
        suave = suave / suave.max() * 255.0
    return suave.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Métricas (opcional)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_metricas(csv_original, csv_clasificado, col_x, col_y,
                      col_gt, col_pred, salida_metricas):
    """
    Clasificación BINARIA: Interior = positivo, Borde = negativo.
    Aisladas excluidas del cálculo.

    Fórmulas (valores en %):
        Recall    = 100 × TP / (TP + FN)
        Precision = 100 × TP / (TP + FP)
        F1        = 2 × Precision × Recall / (Precision + Recall)
        Kappa     = 2×(TP×TN − FN×FP) / [(TP+FP)×(FP+TN) + (TP+FN)×(FN+TN)]

    Args:
        csv_original    (str): CSV con la columna ground truth (col_gt).
        csv_clasificado (str): CSV con la columna de predicciones (col_pred).
        col_x, col_y    (str): Nombres de las columnas de coordenadas.
        col_gt          (str): Columna ground truth en csv_original.
        col_pred        (str): Columna predicción en csv_clasificado.
        salida_metricas (str): Ruta del CSV de salida con las métricas.

    Returns:
        pd.DataFrame: Tabla de métricas.
    """
    print(f"\n[·] Calculando métricas (binario: Interior=+ / Borde=−)...")
    print(f"    GT   → {csv_original}  (columna: '{col_gt}')")
    print(f"    Pred → {csv_clasificado}  (columna: '{col_pred}')")

    for ruta in [csv_original, csv_clasificado]:
        if not Path(ruta).exists():
            sys.exit(f"[ERROR] No se encuentra: {ruta}")

    df_gt   = pd.read_csv(csv_original)
    df_pred = pd.read_csv(csv_clasificado)
    df_gt.columns   = [c.strip() for c in df_gt.columns]
    df_pred.columns = [c.strip() for c in df_pred.columns]

    for col in [col_x, col_y, col_gt]:
        if col not in df_gt.columns:
            sys.exit(f"[ERROR] Columna '{col}' no encontrada en {csv_original}. "
                     f"Disponibles: {list(df_gt.columns)}")
    for col in [col_x, col_y, col_pred]:
        if col not in df_pred.columns:
            sys.exit(f"[ERROR] Columna '{col}' no encontrada en {csv_clasificado}. "
                     f"Disponibles: {list(df_pred.columns)}")

    # Alinear por coordenadas
    df_gt['_x']   = df_gt[col_x].round(1)
    df_gt['_y']   = df_gt[col_y].round(1)
    df_pred['_x'] = df_pred[col_x].round(1)
    df_pred['_y'] = df_pred[col_y].round(1)

    merged = pd.merge(
        df_gt[['_x', '_y', col_gt]],
        df_pred[['_x', '_y', col_pred]],
        on=['_x', '_y'], how='inner'
    )
    print(f"    Partículas alineadas: {len(merged)}")

    # Excluir Aisladas
    merged = merged[
        (merged[col_gt].str.strip().str.lower()   != 'aislada') &
        (merged[col_pred].str.strip().str.lower() != 'aislada')
    ].copy()

    n_eval = len(merged)
    print(f"    Partículas evaluables (sin Aisladas): {n_eval}")

    gt   = merged[col_gt].str.strip().str.capitalize()
    pred = merged[col_pred].str.strip().str.capitalize()

    # ── Tabla de confusión binaria (Interior = positivo) ──────────────────────
    # TP: predicho Interior y era Interior
    # FP: predicho Interior pero era Borde
    # FN: predicho Borde pero era Interior
    # TN: predicho Borde y era Borde
    tp = int(((gt == 'Interior') & (pred == 'Interior')).sum())
    fp = int(((gt == 'Borde')    & (pred == 'Interior')).sum())
    fn = int(((gt == 'Interior') & (pred == 'Borde')).sum())
    tn = int(((gt == 'Borde')    & (pred == 'Borde')).sum())

    # ── Fórmulas de la imagen (valores en %) ─────────────────────────────────
    recall    = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Kappa = 2*(TP*TN - FN*FP) / [(TP+FP)*(FP+TN) + (TP+FN)*(FN+TN)]
    num_k   = 2 * (tp * tn - fn * fp)
    den_k   = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
    kappa   = num_k / den_k if den_k > 0 else 0.0

    accuracy = 100 * (tp + tn) / n_eval if n_eval > 0 else 0.0

    filas = [
        {
            'Metrica':     'TP',
            'Valor':       tp,
            'Descripcion': 'Interior predicho como Interior',
        },
        {
            'Metrica':     'FP',
            'Valor':       fp,
            'Descripcion': 'Borde predicho como Interior',
        },
        {
            'Metrica':     'TN',
            'Valor':       tn,
            'Descripcion': 'Borde predicho como Borde',
        },
        {
            'Metrica':     'FN',
            'Valor':       fn,
            'Descripcion': 'Interior predicho como Borde',
        },
        {
            'Metrica':     'Recall (%)',
            'Valor':       round(recall,    2),
            'Descripcion': '100 × TP / (TP + FN)',
        },
        {
            'Metrica':     'Precision (%)',
            'Valor':       round(precision, 2),
            'Descripcion': '100 × TP / (TP + FP)',
        },
        {
            'Metrica':     'F1 (%)',
            'Valor':       round(f1,        2),
            'Descripcion': '2 × Precision × Recall / (Precision + Recall)',
        },
        {
            'Metrica':     'Kappa',
            'Valor':       round(kappa,     4),
            'Descripcion': '2*(TP*TN−FN*FP) / [(TP+FP)*(FP+TN)+(TP+FN)*(FN+TN)]',
        },
        {
            'Metrica':     'Accuracy (%)',
            'Valor':       round(accuracy,  2),
            'Descripcion': '100 × (TP + TN) / N',
        },
        {
            'Metrica':     'N_evaluadas',
            'Valor':       n_eval,
            'Descripcion': 'Partículas Borde + Interior (sin Aisladas)',
        },
    ]

    df_metricas = pd.DataFrame(filas)

    # ── Imprimir resumen ──────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Tabla de confusión  (Interior=+ / Borde=−)")
    print(f"  {'':<18} Pred Interior  Pred Borde")
    print(f"  {'Real Interior':<18}    {tp:>6}        {fn:>6}")
    print(f"  {'Real Borde':<18}    {fp:>6}        {tn:>6}")
    print(f"{'─'*50}")
    print(f"  Recall    : {recall:.2f} %")
    print(f"  Precision : {precision:.2f} %")
    print(f"  F1        : {f1:.2f} %")
    print(f"  Kappa     : {kappa:.4f}")
    print(f"  Accuracy  : {accuracy:.2f} %")
    print(f"{'─'*50}")
    print(f"{'─'*58}")

    df_metricas.to_csv(salida_metricas, index=False)
    print(f"\n    Métricas guardadas en: {salida_metricas}")

    return df_metricas


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def main(imagen, csv, col_x, col_y, col_label, salida_png, salida_tif,
         dpi, radio_plot, sigma, sin_leyenda,
         calcular_metricas_flag=False,
         csv_gt=None, col_gt=None,
         csv_pred=None, col_pred=None,
         salida_metricas="metricas.csv"):

    for ruta, nombre in [(imagen, "imagen"), (csv, "CSV")]:
        if not Path(ruta).exists():
            sys.exit(f"[ERROR] No se encuentra: {ruta}")

    # ── Imagen ────────────────────────────────────────────────────────────────
    print(f"[·] Cargando imagen: {imagen}")
    blend = cargar_blend(imagen)
    h, w = blend.shape
    print(f"    {w} × {h} px")

    # ── CSV ───────────────────────────────────────────────────────────────────
    print(f"[·] Cargando CSV: {csv}")
    df = pd.read_csv(csv)

    for col in [col_x, col_y, col_label]:
        if col not in df.columns:
            sys.exit(f"[ERROR] Columna '{col}' no encontrada. Disponibles: {list(df.columns)}")

    df["_color"] = df[col_label].apply(asignar_color)

    mask_borde    = df[df["_color"] == COLOR_BORDE]
    mask_aislada  = df[df["_color"] == COLOR_AISLADA]
    mask_interior = df[df["_color"] == COLOR_INTERIOR]

    n_borde    = len(mask_borde)
    n_aislada  = len(mask_aislada)
    n_interior = len(mask_interior)
    print(f"    Borde: {n_borde} | Aislada: {n_aislada} | Interior: {n_interior}")

    # ── PNG de visualización ──────────────────────────────────────────────────
    print(f"[·] Generando PNG: {salida_png}")
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=dpi)
    ax.imshow(blend, cmap="gray", vmin=0, vmax=1, extent=[0, w, h, 0])

    for color, grupo in df.groupby("_color"):
        ax.scatter(grupo[col_x], grupo[col_y],
                   c=color, s=radio_plot ** 2, alpha=0.95,
                   linewidths=0.5, edgecolors="white", zorder=2)

    if not sin_leyenda:
        parches = [
            mpatches.Patch(color=COLOR_BORDE,    label=f"Borde ({n_borde})"),
            mpatches.Patch(color=COLOR_INTERIOR, label=f"Interior ({n_interior})"),
            mpatches.Patch(color=COLOR_AISLADA,  label=f"Aislada ({n_aislada})"),
        ]
        ax.legend(handles=parches, loc="upper right", fontsize=9,
                  framealpha=0.75, edgecolor="white",
                  facecolor="#111111", labelcolor="white")

    nombre_base = Path(csv).stem
    ax.set_title(f"{nombre_base} — Etiquetas de entrenamiento",
                 fontsize=10, color="white", pad=6, backgroundcolor="#222222")
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    fig.savefig(salida_png, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"    Guardado: {salida_png}")

    # ── Máscaras gaussianas ───────────────────────────────────────────────────
    print(f"[·] Generando máscaras gaussianas (sigma={sigma} px)...")

    canal_original = (blend * 255).astype(np.uint8)
    canal_borde    = mascara_gaussiana(h, w, mask_borde[col_x],    mask_borde[col_y],    sigma)
    canal_aislada  = mascara_gaussiana(h, w, mask_aislada[col_x],  mask_aislada[col_y],  sigma)
    canal_interior = mascara_gaussiana(h, w, mask_interior[col_x], mask_interior[col_y], sigma)

    # ── Guardar TIFF como ImageJ hyperstack (C=4, Z=1, T=1) ──────────────────
    stack = np.stack([canal_original, canal_borde, canal_aislada, canal_interior], axis=0)

    print(f"[·] Guardando TIFF hyperstack: {salida_tif}")
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        tmp_path = tmp.name
    tifffile.imwrite(tmp_path, stack, imagej=True, metadata={'axes': 'CYX'})
    shutil.move(tmp_path, salida_tif)
    print(f"    Guardado: {salida_tif}")
    print(f"    Fiji: abre directamente como hyperstack C=4")
    print(f"    → Image > Color > Make Composite")
    print(f"    → Image > Color > Channels Tool (Shift+Z) para asignar colores")

    # ── Métricas (opcional) ───────────────────────────────────────────────────
    if calcular_metricas_flag:
        calcular_metricas(
            csv_original=csv_gt,
            csv_clasificado=csv_pred,
            col_x=col_x,
            col_y=col_y,
            col_gt=col_gt,
            col_pred=col_pred,
            salida_metricas=salida_metricas,
        )

    print("[✓] Listo.")


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros — editar aquí
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Visualización (siempre activo) ────────────────────────────────────────
    imagen      = "../datos/SUb_02_10_merged.tif"
    csv         = "../resultados/resultados_EDT/SUb_02_10_clasificado_EDT.csv"
    col_x       = "X"
    col_y       = "Y"
    col_label   = "clasificacion"   # columna a visualizar en el PNG/TIFF
    salida_png  = "SUb_02_10_clasificado_EDT.png"
    salida_tif  = "SUb_02_10_clasificado_EDT.tif"
    dpi         = 150
    radio_plot  = 6
    sigma       = 2
    sin_leyenda = False

    # ── Métricas (opcional) ───────────────────────────────────────────────────
    # Poner a True para activar el cálculo de métricas
    calcular_metricas_flag = True

    csv_gt          = "../resultados/resultados_EDT/SUb_02_10_clasificado_EDT.csv"          # ground truth
    col_gt          = "Clase"                                         # columna GT
    csv_pred        = "../resultados/resultados_EDT/SUb_02_10_clasificado_EDT.csv"  # predicciones
    col_pred        = "clasificacion"                                 # columna pred
    salida_metricas = "SUb_02_10_metricas_clasificado_EDT.csv"
    # ─────────────────────────────────────────────────────────────────────────

    main(
        imagen=imagen, csv=csv, col_x=col_x, col_y=col_y,
        col_label=col_label, salida_png=salida_png, salida_tif=salida_tif,
        dpi=dpi, radio_plot=radio_plot, sigma=sigma, sin_leyenda=sin_leyenda,
        calcular_metricas_flag=calcular_metricas_flag,
        csv_gt=csv_gt, col_gt=col_gt,
        csv_pred=csv_pred, col_pred=col_pred,
        salida_metricas=salida_metricas,
    )