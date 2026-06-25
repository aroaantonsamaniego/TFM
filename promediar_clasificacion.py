"""
promediar_clasificacion.py
──────────────────────────
Combina las clasificaciones de N runs usando SOFT VOTING: promedia las
probabilidades de clase Interior fila a fila y aplica un umbral óptimo
(el que maximiza F1) sobre el promedio.

Si los CSVs tienen columna 'Clase' (etiqueta real), calcula además:
  - Curvas ROC y PR del ensemble              → SVG
  - Curva Precision / Recall / F1 vs umbral  → SVG
  - Métricas (AUC-ROC, AUC-PR, Precision, Recall, F1, Kappa,
    TP, TN, FP, FN, umbral)                  → CSV

Si no hay etiquetas, genera solo el CSV de clasificación final.

Uso:
    python promediar_clasificacion.py                   # directorio 'clasificado/' por defecto
    python promediar_clasificacion.py mis_resultados/   # directorio indicado
    python promediar_clasificacion.py dir/ --umbral 0.35  # umbral fijo manual
    python promediar_clasificacion.py dir/ --salida resultados_ensemble/
"""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from pathlib import Path
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             cohen_kappa_score, confusion_matrix)


# =========================================================
# CONFIGURACIÓN GLOBAL DE MATPLOTLIB
# (mismo estilo que representaciones.py — editar aquí)
# =========================================================

plt.rcParams['svg.fonttype']        = 'none'
plt.rcParams['lines.linewidth']     = 3
plt.rcParams['legend.frameon']      = False
plt.rcParams['savefig.dpi']         = 300
plt.rcParams['savefig.transparent'] = False
plt.rcParams['savefig.format']      = 'svg'
plt.rc('xtick',  labelsize=16)
plt.rc('ytick',  labelsize=16)
plt.rc('axes',   labelsize=20)
plt.rc('legend', fontsize=16)
plt.rcParams['font.family']         = 'Times New Roman'

# Tamaño figuras cuadradas (ROC, PR — ambos ejes 0→1)
FIG_SIZE      = (9, 9)
# Tamaño figuras con eje X abierto (umbral vs métricas)
FIG_SIZE_WIDE = (12, 7)

# Fracción de positivos en el set de test (línea de referencia en la curva PR)
PR_BASELINE = 0.18


# =========================================================
# NOMBRES DE COLUMNAS EN LOS CSVs DE CADA RUN
# (ajustar si cambian los nombres en el pipeline)
# =========================================================

COL_X             = 'X'
COL_Y             = 'Y'
COL_CLASE         = 'Clase'          # etiqueta real (opcional)
COL_CLASIFICACION = 'clasificacion'  # predicción del modelo
COL_PROB_INT      = 'prob_interior'  # probabilidad de clase Interior

# Mapeo etiqueta real → numérico (Interior/Exterior = positivo = 1)
CLASS_MAP = {'interior': 1, 'exterior': 1, 'borde': 0}


# =========================================================
# UTILIDADES DE ESTILO (réplica de representaciones.py)
# =========================================================

def _save_svg(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format='svg', bbox_inches='tight')
    print(f"  Guardado: {path}")
    plt.close(fig)


def _apply_style(ax, xlabel, ylabel, title,
                 fontsize_title=24, fontsize_label=20,
                 fontsize_ticks=16, grid=True):
    ax.set_xlabel(xlabel, fontsize=fontsize_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    ax.set_title(title,   fontsize=fontsize_title)
    ax.tick_params(labelsize=fontsize_ticks)
    if grid:
        ax.grid(True, alpha=0.3)


def _add_textbox(ax, text, loc='lower right', fontsize=16):
    at = AnchoredText(text, loc=loc,
                      prop=dict(size=fontsize), frameon=True)
    at.patch.set_facecolor('white')
    at.patch.set_alpha(0.9)
    at.patch.set_edgecolor('black')
    ax.add_artist(at)


# =========================================================
# MÉTRICAS
# =========================================================

def calcular_metricas_umbral(y_true, scores, umbral):
    """TP, TN, FP, FN y métricas derivadas para un umbral dado."""
    y_pred = (scores >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    kappa     = cohen_kappa_score(y_true, y_pred)
    return dict(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                precision=precision, recall=recall, f1=f1, kappa=kappa)


def umbral_optimo_f1(y_true, scores):
    """Umbral que maximiza F1 sobre la curva PR."""
    prec, rec, thresh = precision_recall_curve(y_true, scores)
    prec_t, rec_t = prec[:-1], rec[:-1]
    f1 = np.where((prec_t + rec_t) > 0,
                  2 * prec_t * rec_t / (prec_t + rec_t), 0.0)
    return float(thresh[np.argmax(f1)])


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def soft_voting_ensemble(csv_paths: list,
                         output_dir: str = ".",
                         umbral_manual: float = None):
    """
    Combina N CSVs de runs mediante promedio de probabilidades (soft voting).

    Cada CSV debe tener columnas:
      X, Y, clasificacion, prob_interior
      Clase  (opcional — necesaria para calcular métricas y curvas)

    Args:
        csv_paths     (list[str]): Rutas a los CSVs de cada run.
        output_dir    (str):       Directorio de salida.
        umbral_manual (float|None): Umbral fijo; si None se calcula el óptimo por F1.
    """
    if len(csv_paths) < 2:
        raise ValueError("Se necesitan al menos 2 CSVs para el ensemble.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nCargando {len(csv_paths)} runs...")

    # ── 1. Leer CSVs ─────────────────────────────────────────────────────────
    dfs_prob  = []
    clase_ref = None
    tiene_etiquetas = True

    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path, encoding='utf-8-sig')
        # Normalizar nombres de columna para el merge interno
        df_work = df.copy()
        df_work.columns = [c.strip() for c in df_work.columns]

        # Verificar columnas obligatorias
        for col in [COL_X, COL_Y, COL_CLASIFICACION, COL_PROB_INT]:
            if col not in df_work.columns:
                raise ValueError(
                    f"El CSV '{path}' no tiene la columna '{col}'.\n"
                    f"Columnas disponibles: {list(df_work.columns)}")

        df_work['_x'] = df_work[COL_X].round(1)
        df_work['_y'] = df_work[COL_Y].round(1)

        # Probabilidad de Interior (NaN para aisladas)
        df_p = df_work[['_x', '_y', COL_PROB_INT]].rename(
            columns={COL_PROB_INT: f'prob_{i+1}'})
        dfs_prob.append(df_p)

        # Guardar etiquetas reales del primer run (todas deberían ser iguales)
        if COL_CLASE in df_work.columns:
            if clase_ref is None:
                clase_ref = df_work[['_x', '_y', COL_CLASE]].copy()
        else:
            tiene_etiquetas = False

        n_aisl = df_work[COL_CLASIFICACION].str.lower().eq('aislada').sum()
        print(f"  Run {i+1}: {len(df_work)} partículas "
              f"({n_aisl} aisladas excluidas del voto) — {Path(path).name}")

    # ── 2. Merge por posición ─────────────────────────────────────────────────
    merged = dfs_prob[0]
    for df in dfs_prob[1:]:
        merged = pd.merge(merged, df, on=['_x', '_y'], how='inner')

    n_perdidas = len(dfs_prob[0]) - len(merged)
    if n_perdidas > 0:
        print(f"\n  [WARNING] {n_perdidas} partículas no coinciden en todos "
              f"los runs y han sido excluidas.")

    print(f"\n  Partículas comunes a todos los runs: {len(merged)}")

    # Añadir etiquetas reales si existen
    if tiene_etiquetas and clase_ref is not None:
        merged = pd.merge(merged, clase_ref, on=['_x', '_y'], how='inner')

    # ── 3. Soft voting ────────────────────────────────────────────────────────
    prob_cols = [c for c in merged.columns if c.startswith('prob_')]
    merged['prob_interior_media'] = merged[prob_cols].mean(axis=1)

    mask_aisladas = merged['prob_interior_media'].isna()
    mask_clasif   = ~mask_aisladas

    print(f"  Partículas clasificadas (no aisladas): {mask_clasif.sum()}")
    print(f"  Partículas aisladas:                   {mask_aisladas.sum()}")

    scores_clasif = merged.loc[mask_clasif, 'prob_interior_media'].values

    # ── 4. Umbral ─────────────────────────────────────────────────────────────
    if tiene_etiquetas and clase_ref is not None:
        clases_raw = (merged.loc[mask_clasif, COL_CLASE]
                      .str.strip().str.lower())
        y_true_all = np.array([CLASS_MAP.get(c, -1) for c in clases_raw])
        mask_conocidas = y_true_all >= 0
        scores_eval = scores_clasif[mask_conocidas]
        y_true_eval = y_true_all[mask_conocidas]

        if umbral_manual is not None:
            umbral = umbral_manual
            print(f"\n  Umbral manual: {umbral:.4f}")
        else:
            umbral = umbral_optimo_f1(y_true_eval, scores_eval)
            print(f"\n  Umbral óptimo (max F1): {umbral:.4f}")
    else:
        umbral = umbral_manual if umbral_manual is not None else 0.5
        print(f"\n  Sin etiquetas reales — umbral usado: {umbral:.4f}")

    # ── 5. Clasificación final ────────────────────────────────────────────────
    clasificacion = np.where(
        mask_aisladas, 'Aislada',
        np.where(merged['prob_interior_media'] >= umbral,
                 'Interior', 'Borde'))

    # ── 6. CSV de clasificación ───────────────────────────────────────────────
    # Columnas en el mismo orden que los CSVs de cada run:
    # X, Y, Clase (si existe), clasificacion, prob_interior
    cols = {
        COL_X:             merged['_x'].values,
        COL_Y:             merged['_y'].values,
    }
    if tiene_etiquetas and clase_ref is not None:
        cols[COL_CLASE] = merged[COL_CLASE].values
    cols[COL_CLASIFICACION] = clasificacion
    cols[COL_PROB_INT]      = merged['prob_interior_media'].round(4).values

    resultado = pd.DataFrame(cols)

    csv_clasif = out / 'ensemble_clasificacion.csv'
    resultado.to_csv(csv_clasif, index=False)
    print(f"\n  CSV de clasificación guardado en: {csv_clasif}")

    conteo = pd.Series(clasificacion).value_counts()
    print(f"\n  Distribución final:")
    for etiq, n in conteo.items():
        print(f"    {etiq:12s}: {n:4d}  ({100*n/len(clasificacion):.1f} %)")

    # ── 7. Métricas y figuras (solo si hay etiquetas) ─────────────────────────
    if not tiene_etiquetas or clase_ref is None:
        print("\n  Sin columna 'Clase' — se omiten métricas y curvas.")
        return resultado

    # Curvas
    fpr, tpr, _         = roc_curve(y_true_eval, scores_eval)
    auc_roc             = auc(fpr, tpr)
    prec_c, rec_c, th_c = precision_recall_curve(y_true_eval, scores_eval)
    auc_pr              = auc(rec_c, prec_c)

    # F1 vs umbral
    prec_th = prec_c[:-1]
    rec_th  = rec_c[:-1]
    f1_th   = np.where((prec_th + rec_th) > 0,
                       2 * prec_th * rec_th / (prec_th + rec_th), 0.0)

    # Métricas al umbral elegido
    m = calcular_metricas_umbral(y_true_eval, scores_eval, umbral)

    # ── 7a. CSV de métricas ───────────────────────────────────────────────────
    df_met = pd.DataFrame([{
        'umbral':                    round(umbral, 4),
        'AUC_ROC':                   round(auc_roc, 4),
        'AUC_PR':                    round(auc_pr, 4),
        'Precision':                 round(m['precision'], 4),
        'Recall':                    round(m['recall'], 4),
        'F1':                        round(m['f1'], 4),
        'Kappa':                     round(m['kappa'], 4),
        'TP':                        m['tp'],
        'TN':                        m['tn'],
        'FP':                        m['fp'],
        'FN':                        m['fn'],
        'n_runs':                    len(csv_paths),
        'n_particulas_evaluadas':    int(mask_conocidas.sum()),
    }])
    csv_met = out / 'ensemble_metricas.csv'
    df_met.to_csv(csv_met, index=False)
    print(f"\n  Métricas guardadas en: {csv_met}")

    # Resumen consola
    sep = '=' * 52
    print(f"\n{sep}")
    print(f"  MÉTRICAS ENSEMBLE ({len(csv_paths)} runs, soft voting)")
    print(f"{sep}")
    print(f"  Umbral   : {umbral:.4f}"
          + (" (manual)" if umbral_manual else " (max F1)"))
    print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  {'─'*44}")
    print(f"  Recall         : {m['recall']:.4f}")
    print(f"  Precision      : {m['precision']:.4f}")
    print(f"  F1 score       : {m['f1']:.4f}")
    print(f"  Cohen's Kappa  : {m['kappa']:.4f}")
    print(f"  AUC-ROC        : {auc_roc:.4f}")
    print(f"  AUC-PR         : {auc_pr:.4f}")
    print(f"{sep}\n")

    # ── 7b. Curva ROC ─────────────────────────────────────────────────────────
    tpr_u = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
    fpr_u = m['fp'] / (m['fp'] + m['tn']) if (m['fp'] + m['tn']) > 0 else 0

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(fpr, tpr, color='steelblue')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2, alpha=0.8)
    ax.scatter([fpr_u], [tpr_u], color='crimson', zorder=5, s=80,
               label=f'Umbral {umbral:.3f}  '
                     f'(TPR={tpr_u:.2f}, FPR={fpr_u:.2f})')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    _apply_style(ax, 'False Positive Rate', 'True Positive Rate',
                 f'ROC Curve — Ensemble ({len(csv_paths)} runs)')
    _add_textbox(ax, f'AUC = {auc_roc:.3f}', loc='lower right')
    ax.legend(fontsize=14, loc='upper left')
    fig.tight_layout()
    _save_svg(fig, out / 'ensemble_curva_ROC.svg')

    # ── 7c. Curva PR ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(rec_c, prec_c, color='darkorange')
    ax.axhline(PR_BASELINE, linestyle='--', color='gray', lw=2, alpha=0.8,
               label=f'Random classifier  (P = {PR_BASELINE})')
    ax.scatter([m['recall']], [m['precision']], color='crimson', zorder=5, s=80,
               label=f'Umbral {umbral:.3f}  '
                     f'(P={m["precision"]:.2f}, R={m["recall"]:.2f})')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    _apply_style(ax, 'Recall', 'Precision',
                 f'Precision-Recall Curve — Ensemble ({len(csv_paths)} runs)')
    _add_textbox(ax, f'AUC = {auc_pr:.3f}', loc='lower left')
    ax.legend(fontsize=14, loc='upper right')
    fig.tight_layout()
    _save_svg(fig, out / 'ensemble_curva_PR.svg')

    # ── 7d. Precision / Recall / F1 vs Umbral ────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
    ax.plot(th_c, prec_th, color='blue',        label='Precision')
    ax.plot(th_c, rec_th,  color='green',       label='Recall')
    ax.plot(th_c, f1_th,   color='red',         label='F1 Score')
    ax.axvline(umbral, color='red', linestyle='--', lw=2,
               label=f'Opt. threshold = {umbral:.3f}')
    ax.axhline(m['recall'],    color='green', linestyle=':', lw=2)
    ax.axhline(m['precision'], color='blue',  linestyle=':', lw=2)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    _apply_style(ax, 'Threshold', 'Metric Value',
                 f'Metrics vs Threshold — Ensemble ({len(csv_paths)} runs)')
    ax.legend(fontsize=14)
    fig.tight_layout()
    _save_svg(fig, out / 'ensemble_curva_umbral.svg')

    return resultado


# =========================================================
# PUNTO DE ENTRADA
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Ensemble soft voting sobre CSVs de múltiples runs.")
    parser.add_argument(
        "directorio", nargs="?", default="clasificado",
        help="Directorio con los CSVs de cada run (default: 'clasificado/').")
    parser.add_argument(
        "--umbral", type=float, default=None,
        help="Umbral fijo. Si no se indica, se calcula el óptimo por F1.")
    parser.add_argument(
        "--salida", type=str, default=None,
        help="Directorio de salida (default: mismo que el de entrada).")
    args = parser.parse_args()

    directorio = Path(args.directorio)
    if not directorio.exists():
        raise FileNotFoundError(f"No se encuentra el directorio: {directorio}")

    csv_paths = sorted(directorio.glob("*.csv"))
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No se encontraron CSVs en: {directorio}")

    salida = args.salida if args.salida else str(directorio)

    print(f"Directorio : {directorio.absolute()}")
    print(f"Salida     : {Path(salida).absolute()}")
    print(f"CSVs encontrados: {len(csv_paths)}")
    for p in csv_paths:
        print(f"  {p.name}")

    soft_voting_ensemble(
        csv_paths     = [str(p) for p in csv_paths],
        output_dir    = salida,
        umbral_manual = args.umbral,
    )