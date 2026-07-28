"""
promediar_clasificacion.py
──────────────────────────
Combina las clasificaciones de N runs usando SOFT VOTING: promedia las
probabilidades de clase Interior fila a fila y aplica un umbral optimo
(el que maximiza F1) sobre el promedio.

Salidas de clasificacion:
  - ensemble_clasificacion.csv        → todas las imagenes juntas (columna 'imagen')
  - por_imagen/<id_imagen>_ensemble.csv → un CSV por imagen, con el formato de
    entrada de imagenes_clasificado.py (X, Y, Clase, clasificacion, prob_interior)

Si los CSVs tienen columna 'Clase' (etiqueta real), calcula ademas:
  - Curvas ROC y PR del ensemble              → SVG
  - Curva Precision / Recall / F1 vs umbral  → SVG
  - Métricas (AUC-ROC, AUC-PR, Precision, Recall, F1, Kappa,
    TP, TN, FP, FN, umbral)                  → CSV

Si no hay etiquetas, genera solo el CSV de clasificacion final.

Uso:
    python promediar_clasificacion.py                     # usa el cuadro DIRECTORIOS
    python promediar_clasificacion.py mis_resultados/     # sobrescribe la entrada
    python promediar_clasificacion.py dir/ --umbral 0.35  # umbral fijo manual
    python promediar_clasificacion.py dir/ --salida resultados_ensemble/

Los argumentos de consola son opcionales: si no se pasan, se usan los valores
del cuadro de configuración DIRECTORIOS (abajo).
"""

import sys
import re
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



#DIRECTORIOS                          
# Rutas relativas al directorio desde el que se lanza el script, o absolutas.
# Los argumentos de consola, si se pasan, tienen prioridad sobre estos valores.

# Directorio con los CSVs de cada run ('run1_*.csv', 'run2_*.csv', ...)
DIRECTORIO_ENTRADA = "../resultados/early_stopping2/clasificado"

# Directorio donde se guardan CSVs y figuras del ensemble.
# None → mismo directorio que DIRECTORIO_ENTRADA
DIRECTORIO_SALIDA  = None

# Umbral fijo sobre la probabilidad media de Interior.
# None → se calcula el optimo (el que maximiza F1)
UMBRAL_MANUAL      = None

# ═════════════════════════════════════════════════════════


# =========================================================
# CONFIGURACION GLOBAL DE MATPLOTLIB
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
PR_BASELINE = 0.26


# =========================================================
# NOMBRES DE COLUMNAS EN LOS CSVs DE CADA RUN
# =========================================================

COL_X             = 'X'
COL_Y             = 'Y'
COL_CLASE         = 'Clase'          # etiqueta real (opcional)
COL_CLASIFICACION = 'clasificacion'  # prediccion del modelo
COL_PROB_INT      = 'prob_interior'  # probabilidad de clase Interior

# Mapeo etiqueta real - numerico 
CLASS_MAP = {'interior': 1, 'exterior': 1, 'borde': 0}


# =========================================================
# SALIDA DE CSVs POR IMAGEN
# (formato de entrada de imagenes_clasificado.py:
#  X, Y, Clase, clasificacion, prob_interior — un CSV por imagen)
# =========================================================

GUARDAR_CSV_POR_IMAGEN = True          # False → solo el CSV global
SUBDIR_POR_IMAGEN      = 'por_imagen'  # '' o None → mismo directorio de salida
SUFIJO_POR_IMAGEN      = '_ensemble'   # nombre: <id_imagen><SUFIJO>.csv
INCLUIR_COL_IMAGEN     = False         # True → deja la columna 'imagen' también
                                       # en los CSVs por imagen

# Prefijos de los CSVs que genera este mismo script. Se excluyen del glob de
# entrada para que una segunda ejecución sobre el mismo directorio no los
# confunda con CSVs de runs.
PREFIJOS_SALIDA = ('ensemble_',)


# =========================================================
# UTILIDADES DE ESTILO
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
# METRICAS
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
# AGRUPACION DE CSVs POR RUN
# =========================================================

def _run_prefix(nombre: str):
    """
    Extrae el prefijo de run del nombre de archivo: 'run1', 'run2', ...
    Devuelve el prefijo en minúsculas, o None si el archivo no empieza por 'runN'.
    Ej: 'run3_SUb_02_10_orig_clasificado_test.csv' -> 'run3'
    """
    m = re.match(r'^(run\d+)', nombre, flags=re.IGNORECASE)
    return m.group(1).lower() if m else None


def _image_id(nombre: str):
    """
    Identificador de imagen: nombre de archivo sin el prefijo 'runN_' y sin
    extensión. Es idéntico para la misma imagen entre runs distintos, y distinto
    entre imágenes, de modo que sirve como clave para emparejar trayectrias.
    Ej: 'run3_SUb_02_10_orig_clasificado_test.csv' -> 'SUb_02_10_orig_clasificado_test'
    """
    stem = Path(nombre).stem
    return re.sub(r'^run\d+[_-]?', '', stem, flags=re.IGNORECASE)


def _nombre_seguro(texto: str) -> str:
    """
    Sanea un identificador de imagen para usarlo como nombre de archivo:
    sustituye por '_' cualquier carácter no válido en Windows/Linux.
    """
    return re.sub(r'[^\w\-.]', '_', str(texto))


def agrupar_por_run(csv_paths: list):
    """
    Agrupa los CSVs por su prefijo de run ('runN_...'). Todos los archivos que
    comparten el mismo prefijo (es decir, las distintas imágenes de test
    clasificadas por el mismo modelo) se asignan al mismo run.

    Los archivos que no empiecen por 'runN' se tratan como un run independiente
    (usando su nombre completo como clave), preservando el comportamiento previo
    para nomenclaturas antiguas.

    Args:
        csv_paths (list[str]): Rutas a todos los CSVs encontrados.

    Returns:
        dict[str, list[str]]: {nombre_run: [rutas de sus CSVs]}, ordenado de
                              forma natural (run1, run2, ..., run10).
    """
    runs = {}
    for p in csv_paths:
        nombre = Path(p).name
        run = _run_prefix(nombre)
        if run is None:
            run = Path(p).stem  # sin prefijo runN → run propio (compat. antigua)
        runs.setdefault(run, []).append(p)

    def _clave_orden(r):
        m = re.search(r'(\d+)', r)
        return (int(m.group(1)) if m else 0, r)

    return {k: runs[k] for k in sorted(runs, key=_clave_orden)}


# =========================================================
# FUNCION PRINCIPAL
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

    # ── Agrupar los CSVs por prefijo de run (runN_...) ───────────────────────
    # Cada run puede contener varias imágenes de test (varios CSVs). Todas las
    # imagenes de un mismo run se concatenan y el voto se hace ENTRE runs.
    runs = agrupar_por_run(csv_paths)
    run_names = list(runs.keys())
    n_runs    = len(run_names)

    if n_runs < 2:
        raise ValueError(
            f"Se necesitan al menos 2 runs para el ensemble, pero solo se "
            f"detectó {n_runs} a partir de los nombres de archivo: {run_names}.\n"
            f"Comprueba que los CSVs empiecen por 'run1_', 'run2_', ...")

    print(f"\nDetectados {n_runs} runs a partir de los nombres de archivo:")
    for r in run_names:
        imgs = [_image_id(Path(p).name) for p in runs[r]]
        print(f"  {r}: {len(runs[r])} imagen(es) -> {imgs}")

    print(f"\nCargando {n_runs} runs...")

    # ── 1. Leer CSVs, un dataframe por RUN (concatenando sus imágenes) ───────
    dfs_prob  = []
    clase_ref = None
    tiene_etiquetas = True

    for i, run_name in enumerate(run_names):
        piezas     = []
        n_aisl_run = 0

        for path in runs[run_name]:
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

            # Identificador de imagen (nombre sin el prefijo runN_). Distingue
            # trayectorias de imagenes distintas que compartan coordenadas y hace
            # coincidir la misma imagen entre runs.
            df_work['img'] = _image_id(Path(path).name)
            df_work['_x']  = df_work[COL_X].round(1)
            df_work['_y']  = df_work[COL_Y].round(1)

            piezas.append(df_work)
            n_aisl_run += int(df_work[COL_CLASIFICACION].str.lower()
                              .eq('aislada').sum())

        # Todas las imagenes de este run apiladas en un unico dataframe
        df_run = pd.concat(piezas, ignore_index=True)

        # Probabilidad de interior de este run (NaN para aisladas)
        df_p = df_run[['img', '_x', '_y', COL_PROB_INT]].rename(
            columns={COL_PROB_INT: f'prob_{i+1}'})
        dfs_prob.append(df_p)

        # Guardar etiquetas reales del primer run (iguales en todos los runs)
        if COL_CLASE in df_run.columns:
            if clase_ref is None:
                clase_ref = df_run[['img', '_x', '_y', COL_CLASE]].copy()
        else:
            tiene_etiquetas = False

        print(f"  {run_name}: {len(df_run)} trayectorias de "
              f"{len(runs[run_name])} imagen(es) "
              f"({n_aisl_run} aisladas excluidas del voto)")

    # ── 2. Merge por (imagen, posicion) a traves de los runs ─────────────────
    merged = dfs_prob[0]
    for df in dfs_prob[1:]:
        merged = pd.merge(merged, df, on=['img', '_x', '_y'], how='inner')

    n_perdidas = len(dfs_prob[0]) - len(merged)
    if n_perdidas > 0:
        print(f"\n  [WARNING] {n_perdidas} trayectorias no coinciden en todos "
              f"los runs y han sido excluidas.")

    print(f"\n  Taryectorias comunes a todos los runs: {len(merged)}")

    # Anyadir etiquetas reales si existen
    if tiene_etiquetas and clase_ref is not None:
        merged = pd.merge(merged, clase_ref, on=['img', '_x', '_y'], how='inner')

    # ── 3. Soft voting ────────────────────────────────────────────────────────
    prob_cols = [c for c in merged.columns if c.startswith('prob_')]
    merged['prob_interior_media'] = merged[prob_cols].mean(axis=1)

    mask_aisladas = merged['prob_interior_media'].isna()
    mask_clasif   = ~mask_aisladas

    print(f"  Taryectorias clasificadas (no aisladas): {mask_clasif.sum()}")
    print(f"  Taryectorias aisladas:                   {mask_aisladas.sum()}")

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

    # ── 5. Clasificacion final ────────────────────────────────────────────────
    clasificacion = np.where(
        mask_aisladas, 'Aislada',
        np.where(merged['prob_interior_media'] >= umbral,
                 'Interior', 'Borde'))

    # ── 6. CSV de clasificacion ───────────────────────────────────────────────
    # Columnas en el mismo orden que los CSVs de cada run:
    # X, Y, Clase (si existe), clasificacion, prob_interior
    cols = {
        'imagen':          merged['img'].values,
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
    print(f"\n  CSV de clasificación (global) guardado en: {csv_clasif}")

    # ── 6b. Un CSV por imagen (entrada de imagenes_clasificado.py) ────────────
    if GUARDAR_CSV_POR_IMAGEN:
        dir_img = out / SUBDIR_POR_IMAGEN if SUBDIR_POR_IMAGEN else out
        dir_img.mkdir(parents=True, exist_ok=True)

        # Columnas que espera imagenes_clasificado.py (sin 'imagen' por defecto)
        cols_img = [c for c in resultado.columns
                    if INCLUIR_COL_IMAGEN or c != 'imagen']

        print(f"\n  CSVs por imagen en: {dir_img}")
        for img_id, sub in resultado.groupby('imagen', sort=True):
            nombre = f"{_nombre_seguro(img_id)}{SUFIJO_POR_IMAGEN}.csv"
            ruta   = dir_img / nombre
            df_img = sub[cols_img].reset_index(drop=True)
            df_img.to_csv(ruta, index=False)
            print(f"    {nombre:50s} ({len(df_img)} trayectorias)")

    conteo = pd.Series(clasificacion).value_counts()
    print(f"\n  Distribución final:")
    for etiq, n in conteo.items():
        print(f"    {etiq:12s}: {n:4d}  ({100*n/len(clasificacion):.1f} %)")

    # ── 7. Metricas y figuras (solo si hay etiquetas) ─────────────────────────
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

    # Metricas al umbral elegido
    m = calcular_metricas_umbral(y_true_eval, scores_eval, umbral)

    # ── 7a. CSV de metricas ───────────────────────────────────────────────────
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
        'n_runs':                    n_runs,
        'n_particulas_evaluadas':    int(mask_conocidas.sum()),
    }])
    csv_met = out / 'ensemble_metricas.csv'
    df_met.to_csv(csv_met, index=False)
    print(f"\n  Métricas guardadas en: {csv_met}")

    # Resumen consola
    sep = '=' * 52
    print(f"\n{sep}")
    print(f"  MÉTRICAS ENSEMBLE ({n_runs} runs, soft voting)")
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
                 f'ROC Curve — Ensemble ({n_runs} runs)')
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
                 f'Precision-Recall Curve — Ensemble ({n_runs} runs)')
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
                 f'Metrics vs Threshold — Ensemble ({n_runs} runs)')
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
        "directorio", nargs="?", default=None,
        help=f"Directorio con los CSVs de cada run "
             f"(default: '{DIRECTORIO_ENTRADA}', del cuadro DIRECTORIOS).")
    parser.add_argument(
        "--umbral", type=float, default=None,
        help="Umbral fijo. Si no se indica, se calcula el óptimo por F1.")
    parser.add_argument(
        "--salida", type=str, default=None,
        help="Directorio de salida (default: mismo que el de entrada).")
    args = parser.parse_args()

    # Los argumentos de consola, si se pasan, sobrescriben el cuadro DIRECTORIOS
    entrada = args.directorio if args.directorio else DIRECTORIO_ENTRADA
    salida  = args.salida     if args.salida     else DIRECTORIO_SALIDA
    umbral  = args.umbral     if args.umbral is not None else UMBRAL_MANUAL

    directorio = Path(entrada)
    if not directorio.exists():
        raise FileNotFoundError(f"No se encuentra el directorio: {directorio}")

    # Se excluyen los CSVs generados por este mismo script (ejecuciones previas
    # sobre el mismo directorio), que no son runs.
    todos     = sorted(directorio.glob("*.csv"))
    csv_paths = [p for p in todos if not p.name.startswith(PREFIJOS_SALIDA)]
    n_ignorados = len(todos) - len(csv_paths)
    if n_ignorados > 0:
        print(f"[INFO] {n_ignorados} CSV(s) de salida previa ignorados "
              f"(prefijos: {', '.join(PREFIJOS_SALIDA)})")

    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No se encontraron CSVs de run en: {directorio}")

    salida = salida if salida else str(directorio)

    print(f"Directorio : {directorio.absolute()}")
    print(f"Salida     : {Path(salida).absolute()}")
    print(f"CSVs encontrados: {len(csv_paths)}")
    for p in csv_paths:
        print(f"  {p.name}")

    soft_voting_ensemble(
        csv_paths     = [str(p) for p in csv_paths],
        output_dir    = salida,
        umbral_manual = umbral,
    )