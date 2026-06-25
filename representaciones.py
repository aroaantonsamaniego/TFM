import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import os
import glob
from sklearn.metrics import auc
from scipy.interpolate import interp1d

'''
# Todo — comportamiento por defecto (individuales y juntas de los runs)
python representaciones.py

# Solo las figuras combinadas (media de todos los runs)
python representaciones.py --only-combined

# Solo un run concreto
python representaciones.py --run run1

# Varios runs concretos (sus individuales + su combinada entre ellos)
python representaciones.py --run run1 run3 run5
'''



# =========================================================
# CONFIGURACIÓN GLOBAL
# =========================================================

# Directorio con los archivos de cada run.
# El script busca pares: <nombre>_curvas.xlsx  +  <nombre>_metricas.csv
RUNS_DIR = "deep_ensemble/300_epocs/train"

OUTPUT_DIR = "deep_ensemble/300_epocs/figuras"

# Baseline de la curva PR (fracción de positivos en test — siempre igual)
PR_BASELINE_TEST = 0.18

# =========================================================
# PARÁMETROS GLOBALES DE MATPLOTLIB
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
plt.rcParams['font.family'] = 'Times New Roman'

# Tamaño para figuras cuadradas: ROC y PR (ambos ejes 0→1)
FIG_SIZE = (9, 9)
# Tamaño para figuras con eje X abierto: loss, accuracy, umbral
FIG_SIZE_WIDE = (12, 7)

# Paleta de colores — una por run
COLORES = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# =========================================================
# UTILIDADES
# =========================================================

def _save_svg(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.svg")
    fig.savefig(path, format='svg', bbox_inches='tight')
    print(f"  Guardado: {path}")
    plt.close(fig)


def _apply_style(ax, xlabel, ylabel, title, fontsize_title=24,
                 fontsize_label=20, fontsize_ticks=16, grid=True):
    ax.set_xlabel(xlabel, fontsize=fontsize_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    ax.set_title(title, fontsize=fontsize_title)
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


def compute_mean_curve(xy_list):
    """
    Interpola una lista de (x, y) a un eje X común y devuelve
    (common_x, mean_y, std_y, mean_auc).
    Sólo interpola dentro del rango compartido por todas las curvas.
    """
    x_min = max(x[0]  for x, _ in xy_list)
    x_max = min(x[-1] for x, _ in xy_list)
    common_x = np.linspace(x_min, x_max, 500)

    interp_ys = []
    for x, y in xy_list:
        f = interp1d(x, y, kind='linear', bounds_error=False,
                     fill_value=(y[0], y[-1]))
        interp_ys.append(f(common_x))

    interp_ys = np.array(interp_ys)
    aucs = [auc(x, y) for x, y in xy_list]

    return common_x, interp_ys.mean(axis=0), interp_ys.std(axis=0), np.mean(aucs)


# =========================================================
# CARGA DE DATOS
# =========================================================

def load_runs(runs_dir):
    """
    Busca todos los pares <run>_curvas.xlsx + <run>_metricas.csv en runs_dir.
    Devuelve una lista de dicts con todas las hojas y métricas de cada run.
    """
    curvas_files = sorted(glob.glob(os.path.join(runs_dir, "*_curvas.xlsx")))
    if not curvas_files:
        raise FileNotFoundError(
            f"No se encontraron archivos '*_curvas.xlsx' en '{runs_dir}'")

    runs = []
    for curvas_path in curvas_files:
        # Nombre base del run (todo lo que hay antes de "_curvas.xlsx")
        basename   = os.path.basename(curvas_path)
        run_name   = basename.replace("_curvas.xlsx", "")
        metricas_path = curvas_path.replace("_curvas.xlsx", "_metricas.csv")

        if not os.path.exists(metricas_path):
            print(f"  [AVISO] Sin CSV de métricas para '{run_name}', se omite.")
            continue

        # ── Métricas del CSV ──────────────────────────────────────────────────
        df_meta = pd.read_csv(metricas_path)
        row_val  = df_meta[df_meta['tipo'] == 'mejor_modelo_val'].iloc[0]
        row_test = df_meta[df_meta['tipo'] == 'test'].iloc[0]

        meta = {
            'run_name':      run_name,
            'best_epoch':    int(row_val['epoch']),
            # Validación
            'auc_roc_val':   float(row_val['auc_roc']),
            'auc_pr_val':    float(row_val['auc_pr']),
            'umbral_val':    float(row_val['umbral_optimo']),
            'recall_val':    float(row_val['recall']),
            'precision_val': float(row_val['precision']),
            'f1_val':        float(row_val['f1']),
            'kappa_val':     float(row_val['kappa']),
            # Test
            'auc_roc_test':  float(row_test['auc_roc']),
            'auc_pr_test':   float(row_test['auc_pr']),
            'umbral_test':   float(row_test['umbral_optimo']),
            'recall_test':   float(row_test['recall']),
            'precision_test':float(row_test['precision']),
            'f1_test':       float(row_test['f1']),
            'kappa_test':    float(row_test['kappa']),
        }

        # ── Hojas del Excel ───────────────────────────────────────────────────
        xl = pd.ExcelFile(curvas_path)
        sheets = {}
        for s in xl.sheet_names:
            sheets[s] = pd.read_excel(curvas_path, sheet_name=s)

        runs.append({'meta': meta, 'sheets': sheets})
        print(f"  Run cargado: {run_name}  "
              f"(val AUC-ROC={meta['auc_roc_val']:.4f} | "
              f"test AUC-ROC={meta['auc_roc_test']:.4f})")

    print(f"\n  Total: {len(runs)} run/s cargados.\n")
    return runs


# =========================================================
# FIGURAS POR RUN: pérdidas y umbral
# =========================================================

def plot_loss_acc(runs):
    """
    Una figura de loss y una de accuracy por run.
    Si hay varios runs, también genera una figura combinada por métrica.
    """
    # ── Individuales ──────────────────────────────────────────────────────────
    for run in runs:
        if run.get('_skip_individual'):
            continue
        df   = run['sheets']['Perdidas_y_accuracy']
        name = run['meta']['run_name']

        # Loss
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        ax.plot(df['Epoch'], df['Train_Loss'], color='red',  label='Train Loss')
        ax.plot(df['Epoch'], df['Val_Loss'],   color='blue', label='Validation Loss')
        _apply_style(ax, 'Epoch', 'Loss', 'Training & Validation Loss')
        ax.legend(fontsize=16)
        fig.tight_layout()
        _save_svg(fig, f"{name}_loss")

        # Accuracy
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        ax.plot(df['Epoch'], df['Train_Acc'], color='red',  label='Train Accuracy')
        ax.plot(df['Epoch'], df['Val_Acc'],   color='blue', label='Validation Accuracy')
        _apply_style(ax, 'Epoch', 'Accuracy (%)', 'Training & Validation Accuracy')
        ax.legend(fontsize=16)
        fig.tight_layout()
        _save_svg(fig, f"{name}_accuracy")

    # ── Combinadas (sólo si hay más de un run) ────────────────────────────────
    if len(runs) > 1:
        for col, ylabel, title, fname in [
            ('Train_Loss', 'Loss',          'Training Loss — All Runs',      'all_train_loss'),
            ('Val_Loss',   'Loss',          'Validation Loss — All Runs',    'all_val_loss'),
            ('Train_Acc',  'Accuracy (%)',  'Training Accuracy — All Runs',  'all_train_acc'),
            ('Val_Acc',    'Accuracy (%)',  'Validation Accuracy — All Runs','all_val_acc'),
        ]:
            fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
            for i, run in enumerate(runs):
                df   = run['sheets']['Perdidas_y_accuracy']
                name = run['meta']['run_name']
                ax.plot(df['Epoch'], df[col],
                        color=COLORES[i % len(COLORES)], label=name)
            _apply_style(ax, 'Epoch', ylabel, title)
            ax.legend(fontsize=16)
            fig.tight_layout()
            _save_svg(fig, fname)


def plot_umbral(runs):
    """
    Curva umbral vs Recall / Precision / F1 — SÓLO TEST.
    Una figura individual por run, más una combinada si hay varios.
    El umbral óptimo del modelo se marca con línea vertical,
    y los valores de recall y precision óptimos con líneas horizontales.
    """
    # ── Individuales ──────────────────────────────────────────────────────────
    for run in runs:
        if run.get('_skip_individual'):
            continue
        sheet = 'Curva_umbral_test'
        if sheet not in run['sheets']:
            print(f"  [AVISO] {run['meta']['run_name']}: sin hoja '{sheet}', se omite.")
            continue

        df   = run['sheets'][sheet]
        meta = run['meta']
        name = meta['run_name']

        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        ax.plot(df['Umbral'], df['Recall'],    color='green', label='Recall')
        ax.plot(df['Umbral'], df['Precision'], color='blue',  label='Precision')
        ax.plot(df['Umbral'], df['F1'],        color='red',   label='F1 Score')

        ax.axvline(meta['umbral_test'], color='red',   linestyle='--', linewidth=2,
                   label=f"Opt. threshold = {meta['umbral_test']:.3f}")
        ax.axhline(meta['recall_test'],    color='green', linestyle=':', linewidth=2)
        ax.axhline(meta['precision_test'], color='blue',  linestyle=':', linewidth=2)

        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        _apply_style(ax, 'Threshold', 'Metric Value',
                     'Metrics vs Threshold — Test')
        ax.legend(fontsize=14)
        fig.tight_layout()
        _save_svg(fig, f"{name}_umbral_test")

    # ── Combinada ─────────────────────────────────────────────────────────────
    if len(runs) > 1:
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        for i, run in enumerate(runs):
            sheet = 'Curva_umbral_test'
            if sheet not in run['sheets']:
                continue
            df   = run['sheets'][sheet]
            name = run['meta']['run_name']
            # Sólo F1 en la combinada para no saturar
            ax.plot(df['Umbral'], df['F1'],
                    color=COLORES[i % len(COLORES)], label=f"{name} — F1")

        _apply_style(ax, 'Threshold', 'F1 Score',
                     'F1 vs Threshold — Test (All Runs)')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.legend(fontsize=14)
        fig.tight_layout()
        _save_svg(fig, 'all_f1_umbral_test')


# =========================================================
# FIGURAS DE CURVAS ROC Y PR — TEST
# =========================================================

def _plot_single_roc_test(run):
    """ROC de test para un único run con textbox de AUC."""
    sheet = 'Curva_ROC_test'
    if sheet not in run['sheets']:
        return
    df   = run['sheets'][sheet]
    meta = run['meta']
    name = meta['run_name']

    x = df['FPR'].values
    y = df['TPR'].values

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(x, y, color='steelblue')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, alpha=0.8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    _apply_style(ax, 'False Positive Rate', 'True Positive Rate',
                 'ROC Curve — Test')
    _add_textbox(ax, f"AUC = {meta['auc_roc_test']:.3f}", loc='lower right')
    fig.tight_layout()
    _save_svg(fig, f"{name}_ROC_test")


def _plot_single_pr_test(run):
    """PR de test para un único run con textbox de AUC y baseline fija."""
    sheet = 'Curva_PR_test'
    if sheet not in run['sheets']:
        return
    df   = run['sheets'][sheet]
    meta = run['meta']
    name = meta['run_name']

    x = df['Recall'].values
    y = df['Precision'].values

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(x, y, color='darkorange')
    ax.axhline(y=PR_BASELINE_TEST, linestyle='--', color='gray',
               linewidth=2, alpha=0.8,
               label=f"Random classifier (P = {PR_BASELINE_TEST})")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    _apply_style(ax, 'Recall', 'Precision',
                 'Precision-Recall Curve — Test')
    _add_textbox(ax, f"AUC = {meta['auc_pr_test']:.3f}", loc='lower left')
    ax.legend(fontsize=14, loc='upper right')
    fig.tight_layout()
    _save_svg(fig, f"{name}_PR_test")


def plot_roc_pr_test(runs):
    """
    Genera:
      - ROC test individual por run
      - PR  test individual por run
      - ROC test combinada con media ± std  (sólo si hay > 1 run)
      - PR  test combinada con media ± std  (sólo si hay > 1 run)
    """
    for run in runs:
        if run.get('_skip_individual'):
            continue
        _plot_single_roc_test(run)
        _plot_single_pr_test(run)

    if len(runs) <= 1:
        return

    # ── ROC combinada ─────────────────────────────────────────────────────────
    roc_xy = []
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for i, run in enumerate(runs):
        sheet = 'Curva_ROC_test'
        if sheet not in run['sheets']:
            continue
        df   = run['sheets'][sheet]
        meta = run['meta']
        x = df['FPR'].values
        y = df['TPR'].values
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        roc_xy.append((x, y))
        ax.plot(x, y, color=COLORES[i % len(COLORES)], alpha=0.85,
                label=f"{meta['run_name']}  (AUC = {meta['auc_roc_test']:.4f})")

    if roc_xy:
        cx, my, _, mauc = compute_mean_curve(roc_xy)
        ax.plot(cx, my, color='black', lw=3,
                label=f"Mean  (AUC = {mauc:.4f})")

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2, alpha=0.8,
            label="Random (AUC = 0.5)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    _apply_style(ax, 'False Positive Rate', 'True Positive Rate',
                 'ROC Curves — Test')
    ax.legend(fontsize=14)
    fig.tight_layout()
    _save_svg(fig, 'all_ROC_test')

    # ── PR combinada ──────────────────────────────────────────────────────────
    pr_xy = []
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for i, run in enumerate(runs):
        sheet = 'Curva_PR_test'
        if sheet not in run['sheets']:
            continue
        df   = run['sheets'][sheet]
        meta = run['meta']
        x = df['Recall'].values
        y = df['Precision'].values
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        pr_xy.append((x, y))
        ax.plot(x, y, color=COLORES[i % len(COLORES)], alpha=0.85,
                label=f"{meta['run_name']}  (AUC = {meta['auc_pr_test']:.4f})")

    if pr_xy:
        cx, my, _, mauc = compute_mean_curve(pr_xy)
        ax.plot(cx, my, color='black', lw=3,
                label=f"Mean  (AUC = {mauc:.4f})")

    ax.axhline(y=PR_BASELINE_TEST, linestyle='--', color='gray',
               linewidth=2, alpha=0.8,
               label=f"Random classifier (P = {PR_BASELINE_TEST})")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    _apply_style(ax, 'Recall', 'Precision',
                 'Precision-Recall Curves — Test')
    ax.legend(fontsize=14)
    fig.tight_layout()
    _save_svg(fig, 'all_PR_test')


# =========================================================
# TABLA DE MÉTRICAS POR CONSOLA
# =========================================================

def print_metrics_table(runs):
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  MÉTRICAS TEST — RESUMEN DE RUNS")
    print(f"{sep}")
    print(f"  {'Run':<18} {'AUC-ROC':>8} {'AUC-PR':>8} {'Umbral':>8} "
          f"{'Recall':>8} {'Prec':>8} {'F1':>8} {'Kappa':>8}")
    print(f"  {'─'*82}")
    for run in runs:
        m = run['meta']
        print(f"  {m['run_name']:<18} "
              f"{m['auc_roc_test']:>8.4f} "
              f"{m['auc_pr_test']:>8.4f} "
              f"{m['umbral_test']:>8.4f} "
              f"{m['recall_test']:>8.4f} "
              f"{m['precision_test']:>8.4f} "
              f"{m['f1_test']:>8.4f} "
              f"{m['kappa_test']:>8.4f}")
    print(f"{sep}\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Genera figuras SVG a partir de los runs de entrenamiento.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Ejemplos de uso:\n"
            "  python representar_unificado.py                  # todo\n"
            "  python representar_unificado.py --only-combined  # solo figuras combinadas\n"
            "  python representar_unificado.py --run run1       # solo el run 'run1'\n"
            "  python representar_unificado.py --run run1 run2  # varios runs concretos\n"
        )
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run", nargs="+", metavar="NOMBRE",
        help="Representa únicamente el/los run/s indicados (por nombre, sin extensión)."
    )
    mode.add_argument(
        "--only-combined", action="store_true",
        help="Representa únicamente las figuras combinadas de todos los runs."
    )

    args = parser.parse_args()

    # ── Cargar todos los runs disponibles ────────────────────────────────────
    print(f"\nBuscando runs en: '{RUNS_DIR}/'")
    all_runs = load_runs(RUNS_DIR)
    available = [r['meta']['run_name'] for r in all_runs]

    # ── Seleccionar el subconjunto a representar ──────────────────────────────
    if args.run:
        # Validar nombres pedidos
        not_found = [n for n in args.run if n not in available]
        if not_found:
            parser.error(
                f"Run/s no encontrado/s: {not_found}\n"
                f"Disponibles: {available}"
            )
        runs_to_plot = [r for r in all_runs if r['meta']['run_name'] in args.run]
        combined     = False
        print(f"\nModo: run/s concreto/s → {[r['meta']['run_name'] for r in runs_to_plot]}")

    elif args.only_combined:
        runs_to_plot = all_runs   # necesitamos todos para calcular la media
        combined     = True
        print(f"\nModo: solo figuras combinadas ({len(all_runs)} runs)")

    else:
        runs_to_plot = all_runs
        combined     = False
        print(f"\nModo: todo ({len(all_runs)} runs)")

    # ── Generar figuras ───────────────────────────────────────────────────────

    if args.only_combined:
        # Solo las figuras combinadas (requiere >1 run)
        if len(all_runs) < 2:
            print("  [AVISO] Solo hay 1 run cargado: las figuras combinadas no se generan.")
        else:
            for r in all_runs:
                r['_skip_individual'] = True

            print("── Generando figuras combinadas de pérdida y accuracy ────")
            plot_loss_acc(all_runs)

            print("── Generando figura combinada de umbral (test) ───────────")
            plot_umbral(all_runs)

            print("── Generando figuras combinadas ROC y PR (test) ──────────")
            plot_roc_pr_test(all_runs)

            for r in all_runs:
                r.pop('_skip_individual', None)
    else:
        print("── Generando figuras de pérdida y accuracy ──────────────────")
        plot_loss_acc(runs_to_plot)

        print("── Generando curvas umbral (test) ───────────────────────────")
        plot_umbral(runs_to_plot)

        print("── Generando curvas ROC y PR (test) ─────────────────────────")
        plot_roc_pr_test(runs_to_plot)

    print_metrics_table(runs_to_plot)
    print(f"\nFiguras guardadas en: {OUTPUT_DIR}/")