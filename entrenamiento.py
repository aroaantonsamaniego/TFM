import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from modelo_CNN import MitochondriaContextCNN
from funciones_auxiliares import build_dataset_from_csv, load_pairs_from_dir, visualize_loader

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("entrenamiento.log", mode='w')]
)
logger = logging.getLogger(__name__)


def prepare_loaders(tif_path, csv_path, patch_size=64, batch_size=32, val_split=0.2):
    '''
    Construye los DataLoaders de entrenamiento y validación.
    Las partículas etiquetadas como 'aislada' se eliminan automáticamente
    dentro de build_dataset_from_csv antes de construir los loaders.

    Acepta rutas individuales (str) o listas de rutas (list).

    Args:
        tif_path   (str | list[str]): Ruta/s al .tif de 2 canales.
        csv_path   (str | list[str]): Ruta/s al CSV con columnas y, x, clase.
        patch_size (int):             Tamaño del recorte (default: 64).
        batch_size (int):             Tamaño del batch (default: 32).
        val_split  (float):           Fracción para validación (default: 0.2).

    Returns:
        tuple: (train_loader, val_loader)
    '''
    n_fuentes = len(tif_path) if isinstance(tif_path, list) else 1
    logger.info(f"Cargando datos de {n_fuentes} fuente/s "
                f"(las partículas 'aislada' se excluyen automáticamente)...")

    dataset  = build_dataset_from_csv(tif_path, csv_path, patch_size)
    val_size = int(len(dataset) * val_split)
    trn_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [trn_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    logger.info(f"Train: {trn_size} muestras | Val: {val_size} muestras")
    return train_loader, val_loader


def calcular_metricas(val_scores, val_labels):
    '''
    Calcula TP, TN, FP, FN al umbral óptimo (el que maximiza F1) y devuelve
    todas las métricas de la imagen: Recall, Precision, F1 y Cohen's Kappa.

    Args:
        val_scores (list of float): Probabilidades de clase Interior (clase positiva).
        val_labels (list of int):   Etiquetas reales (0=Borde, 1=Interior).

    Returns:
        dict con claves: TP, TN, FP, FN, recall, precision, f1, kappa, umbral_optimo.
    '''
    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)

    # Buscar el umbral que maximiza F1 sobre la curva PR
    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_scores)
    # precisions y recalls tienen un elemento más que thresholds
    f1s = np.where((precisions[:-1] + recalls[:-1]) > 0,
                   2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
                   0)
    idx_opt       = int(np.argmax(f1s)) #sacamos posicion del array donde esta el f1 max
    umbral_optimo = float(thresholds[idx_opt]) #cogemos el umbral que maximiza f1

    # Calcular TP, TN, FP, FN con ese umbral
    preds = (val_scores >= umbral_optimo).astype(int)
    TP = int(((preds == 1) & (val_labels == 1)).sum()) #el operador & internamente hace +1 si se cumplen ambas condiciones
    TN = int(((preds == 0) & (val_labels == 0)).sum())
    FP = int(((preds == 1) & (val_labels == 0)).sum())
    FN = int(((preds == 0) & (val_labels == 1)).sum())

    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    kappa_num = 2 * (TP * TN - FN * FP)
    kappa_den = (TP + FP) * (FP + TN) + (TP + FN) * (FN + TN)
    kappa     = kappa_num / kappa_den if kappa_den > 0 else 0.0

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'recall':    recall,
        'precision': precision,
        'f1':        f1,
        'kappa':     kappa,
        'umbral_optimo': umbral_optimo,
    }


def calcular_metricas_con_umbral(scores, labels, umbral):
    '''
    Igual que calcular_metricas pero usando un umbral ya conocido en lugar de
    buscar el óptimo. Se usa para el set de test: las métricas deben calcularse
    con el umbral guardado durante el entrenamiento (el mismo que usa la red
    para clasificar), no con el que maximiza F1 sobre los datos de test.

    Args:
        scores (array-like): Probabilidades de clase Interior.
        labels (array-like): Etiquetas reales (0=Borde, 1=Interior).
        umbral (float):      Umbral de decisión a aplicar.

    Returns:
        dict con claves: TP, TN, FP, FN, recall, precision, f1, kappa,
                         umbral_optimo (igual al umbral pasado).
    '''
    scores = np.array(scores)
    labels = np.array(labels)

    preds = (scores >= umbral).astype(int)
    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())

    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    kappa_num = 2 * (TP * TN - FN * FP)
    kappa_den = (TP + FP) * (FP + TN) + (TP + FN) * (FN + TN)
    kappa     = kappa_num / kappa_den if kappa_den > 0 else 0.0

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'recall':        recall,
        'precision':     precision,
        'f1':            f1,
        'kappa':         kappa,
        'umbral_optimo': umbral,  # el umbral usado, no buscado
    }


def plot_curvas_evaluacion(val_scores, val_labels, save_dir=None, prefijo='val',
                           umbral_externo=None):
    '''
    Calcula todas las métricas de evaluación del mejor modelo, las imprime
    por consola y las registra en el log. Genera dos figuras independientes:
      - Curva ROC (FPR vs TPR) con AUC en la leyenda.
      - Curva PR  (Recall vs Precision) con Average Precision en la leyenda.
      - Curva Umbral vs Recall / Precision / F1.

    El criterio de guardado del modelo sigue siendo exclusivamente el AUC-ROC.
    Las métricas adicionales (Recall, Precision, F1, Kappa) se calculan al
    umbral óptimo que maximiza F1, solo a efectos informativos.
    Las métricas NO se muestran dentro de las gráficas — solo por consola y log.

    Args:
        val_scores     (list of float): Probabilidades de clase Interior (validación).
        val_labels     (list of int):   Etiquetas reales (0=Borde, 1=Interior).
        save_dir       (str | None):    Directorio donde guardar las figuras.
        umbral_externo (float | None):  Si se indica, las métricas se calculan con
                                        este umbral en lugar de buscar el óptimo.
                                        Usar siempre para test (umbral del modelo).

    Returns:
        float: AUC-ROC (el criterio de guardado del modelo).
    '''
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)

    # ── Curvas ────────────────────────────────────────────────────────────────
    fpr, tpr, _            = roc_curve(val_labels, val_scores)
    roc_auc                = auc(fpr, tpr)
    precisions, recalls, _ = precision_recall_curve(val_labels, val_scores)
    pr_auc                 = auc(recalls, precisions)

    # ── Métricas ──────────────────────────────────────────────────────────────
    # Test: usar el umbral del modelo guardado, NO buscar el óptimo sobre test
    # Validación: buscar el umbral que maximiza F1 sobre validación
    if umbral_externo is not None:
        m = calcular_metricas_con_umbral(val_scores, val_labels, umbral_externo)
    else:
        m = calcular_metricas(val_scores, val_labels)

    # ── Consola ───────────────────────────────────────────────────────────────
    etiq = 'VALIDACIÓN — MEJOR MODELO' if prefijo == 'val' else 'TEST'
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  MÉTRICAS DE {etiq}")
    print(f"{sep}")
    etiq_umbral = 'Umbral del modelo     ' if umbral_externo is not None else 'Umbral óptimo (max F1)'
    print(f"  {etiq_umbral} : {m['umbral_optimo']:.4f}")
    print(f"  TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")
    print(f"  {'─'*46}")
    print(f"  Recall         : {m['recall']:.4f}")
    print(f"  Precision      : {m['precision']:.4f}")
    print(f"  F1 score       : {m['f1']:.4f}")
    print(f"  Cohen's Kappa  : {m['kappa']:.4f}")
    print(f"  AUC-ROC        : {roc_auc:.4f}")
    print(f"  AUC-PR         : {pr_auc:.4f}")
    print(f"{sep}\n")

    # ── Log ───────────────────────────────────────────────────────────────────
    logger.info(f"── Métricas {etiq} ──")
    logger.info(f"  AUC-ROC        : {roc_auc:.4f}  (criterio de guardado)")
    etiq_umbral_log = 'Umbral modelo' if umbral_externo is not None else 'Umbral opt.'
    logger.info(f"  {etiq_umbral_log}    : {m['umbral_optimo']:.4f}  "
                f"({'del modelo guardado' if umbral_externo is not None else 'maximiza F1'})")
    logger.info(f"  TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")
    logger.info(f"  Recall         : {m['recall']:.4f}")
    logger.info(f"  Precision      : {m['precision']:.4f}")
    logger.info(f"  F1 score       : {m['f1']:.4f}")
    logger.info(f"  Cohen's Kappa  : {m['kappa']:.4f}")
    logger.info(f"  AUC-PR         : {pr_auc:.4f}")

    # ── Rutas de guardado ─────────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = "resultados_entrenamiento"
        os.makedirs(save_dir, exist_ok=True)
    titulo_roc = (f'Curva ROC — Validación del mejor modelo\nClasificador Borde / Interior'
                  if prefijo == 'val' else
                  f'Curva ROC — Set de Test\nClasificador Borde / Interior')
    titulo_pr  = (f'Curva PR — Validación del mejor modelo\nClasificador Borde / Interior'
                  if prefijo == 'val' else
                  f'Curva PR — Set de Test\nClasificador Borde / Interior')
    path_roc    = os.path.join(save_dir, f"curva_ROC_{prefijo}.png")
    path_pr     = os.path.join(save_dir, f"curva_PR_{prefijo}.png")
    path_umbral = os.path.join(save_dir, f"curva_umbral_{prefijo}.png")

    # ── Curva ROC ─────────────────────────────────────────────────────────────
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    ax_roc.plot(fpr, tpr, color='steelblue', lw=2,
                label=f'Curva ROC  (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=1.2,
                linestyle='--', label='Clasificador aleatorio (AUC = 0.5)')
    ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    ax_roc.set_title(titulo_roc, fontsize=13)
    ax_roc.legend(loc='lower right', fontsize=11)
    ax_roc.grid(True, alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(path_roc, dpi=150, bbox_inches='tight')
    plt.close(fig_roc)
    print(f"  Curva ROC guardada en: {path_roc}")

    # ── Curva PR ──────────────────────────────────────────────────────────────
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    ax_pr.plot(recalls, precisions, color='darkorange', lw=2,
               label=f'Curva PR  (AUC-PR = {pr_auc:.3f})')
    baseline = val_labels.mean()
    ax_pr.axhline(y=baseline, color='gray', lw=1.2, linestyle='--',
                  label=f'Clasificador aleatorio (P = {baseline:.2f})')
    ax_pr.set_xlim([0.0, 1.0]); ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall (Sensibilidad)', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title(titulo_pr, fontsize=13)
    ax_pr.legend(loc='upper right', fontsize=11)
    ax_pr.grid(True, alpha=0.3)
    fig_pr.tight_layout()
    fig_pr.savefig(path_pr, dpi=150, bbox_inches='tight')
    plt.close(fig_pr)
    print(f"  Curva PR  guardada en: {path_pr}")

    # ── Curva Umbral vs Recall / Precision / F1 ───────────────────────────────
    # Permite elegir visualmente el umbral de decisión según la prioridad:
    # más recall (menos FN) o más precision (menos FP).
    prec_th, rec_th, thresholds_th = precision_recall_curve(val_labels, val_scores)
    # precision_recall_curve devuelve un punto extra al final → recortamos
    prec_th = prec_th[:-1]
    rec_th  = rec_th[:-1]
    f1_th   = np.where((prec_th + rec_th) > 0,
                       2 * prec_th * rec_th / (prec_th + rec_th), 0)

    fig_u, ax_u = plt.subplots(figsize=(8, 5))
    ax_u.plot(thresholds_th, rec_th,  color='steelblue',  lw=2, label='Recall (↑ = menos FN)')
    ax_u.plot(thresholds_th, prec_th, color='darkorange',  lw=2, label='Precision (↑ = menos FP)')
    ax_u.plot(thresholds_th, f1_th,   color='seagreen',    lw=2, label='F1')
    ax_u.axvline(m['umbral_optimo'], color='gray', linestyle='--', alpha=0.8,
                 label=f'Umbral óptimo F1 = {m["umbral_optimo"]:.3f}')
    ax_u.axhline(m['recall'],    color='steelblue',  linestyle=':', alpha=0.5)
    ax_u.axhline(m['precision'], color='darkorange', linestyle=':', alpha=0.5)
    ax_u.set_xlim([0.0, 1.0]); ax_u.set_ylim([0.0, 1.05])
    ax_u.set_xlabel('Umbral de decisión (prob. Interior)', fontsize=12)
    ax_u.set_ylabel('Valor de la métrica', fontsize=12)
    ax_u.set_title('Umbral vs Recall / Precision / F1\n'
                   'Mueve el umbral a la izquierda para más Recall (menos FN)', fontsize=12)
    ax_u.legend(loc='center right', fontsize=10)
    ax_u.grid(True, alpha=0.3)
    fig_u.tight_layout()
    fig_u.savefig(path_umbral, dpi=150, bbox_inches='tight')
    plt.close(fig_u)
    print(f"  Curva umbral  guardada en: {path_umbral}")

    return roc_auc


def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None, save_dir=None):
    """
    Representa las curvas de pérdida y, si se proporcionan, las de accuracy.
    Genera figuras separadas para loss y accuracy.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    epochs = range(1, len(train_losses) + 1)

    # ── Figura 1: pérdidas ────────────────────────────────────────────────────
    fig_l, ax_l = plt.subplots(figsize=(8, 5))
    ax_l.plot(epochs, train_losses, label='Train Loss', color='steelblue', linewidth=2)
    ax_l.plot(epochs, val_losses,   label='Val Loss',   color='tomato',    linewidth=2)
    best_epoch_l = int(np.argmin(val_losses)) + 1
    ax_l.axvline(best_epoch_l, color='gray', linestyle='--', alpha=0.7,
                 label=f'Mejor val_loss (epoch {best_epoch_l})')
    ax_l.set_xlabel('Epoch'); ax_l.set_ylabel('Loss')
    ax_l.set_title('Curvas de pérdida')
    ax_l.legend(); ax_l.grid(True, alpha=0.3)
    fig_l.tight_layout()

    # ── Figura 2: accuracy (solo si se proporcionan) ──────────────────────────
    fig_a = None
    if train_accs is not None and val_accs is not None:
        fig_a, ax_a = plt.subplots(figsize=(8, 5))
        ax_a.plot(epochs, train_accs, label='Train Accuracy', color='steelblue', linewidth=2)
        ax_a.plot(epochs, val_accs,   label='Val Accuracy',   color='tomato',    linewidth=2)
        best_epoch_a = int(np.argmax(val_accs)) + 1
        ax_a.axvline(best_epoch_a, color='gray', linestyle='--', alpha=0.7,
                     label=f'Mejor val_acc (epoch {best_epoch_a})')
        ax_a.set_xlabel('Epoch'); ax_a.set_ylabel('Accuracy (%)')
        ax_a.set_title('Curvas de accuracy')
        ax_a.legend(); ax_a.grid(True, alpha=0.3)
        fig_a.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path_l = os.path.join(save_dir, 'training_curves_loss.png')
        fig_l.savefig(path_l, dpi=150, bbox_inches='tight')
        logger.info(f"Curvas loss guardadas en {path_l}")
        print(f"  Curvas loss     guardadas en: {path_l}")
        if fig_a is not None:
            path_a = os.path.join(save_dir, 'training_curves_accuracy.png')
            fig_a.savefig(path_a, dpi=150, bbox_inches='tight')
            logger.info(f"Curvas accuracy guardadas en {path_a}")
            print(f"  Curvas accuracy guardadas en: {path_a}")
    plt.close(fig_l)
    if fig_a is not None:
        plt.close(fig_a)


def plot_curvas_por_epoch(historial_curvas, best_epoch_num, best_val_scores,
                           best_val_labels, save_dir=None):
    """
    Genera dos figuras multiepoch (ROC y PR) con una curva por cada checkpoint
    guardado cada 5 epochs a partir de la epoch 30, más la curva del mejor modelo
    resaltada en negro. Los plots individuales del mejor modelo (curva_ROC.png y
    curva_PR.png) se generan aparte en plot_curvas_evaluacion y NO se tocan aquí.

    Args:
        historial_curvas (list of dict): Cada dict tiene:
            'epoch'  (int)   : número de epoch
            'fpr'    (array) : false positive rate
            'tpr'    (array) : true positive rate
            'roc_auc'(float) : AUC-ROC
            'prec'   (array) : precision
            'rec'    (array) : recall
            'pr_auc' (float) : AUC-PR
        best_epoch_num   (int)   : epoch del mejor modelo guardado.
        best_val_scores  (list)  : scores del mejor modelo.
        best_val_labels  (list)  : etiquetas del mejor modelo.
        save_dir         (str|None): directorio donde guardar las figuras.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve

    if not historial_curvas:
        return

    if save_dir is None:
        save_dir = "resultados_entrenamiento"
    os.makedirs(save_dir, exist_ok=True)

    # Paleta de colores para las curvas por epoch
    cmap    = plt.cm.plasma
    n       = len(historial_curvas)
    colores = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # Curvas del mejor modelo
    bvs = np.array(best_val_scores)
    bvl = np.array(best_val_labels)
    fpr_best, tpr_best, _ = roc_curve(bvl, bvs)
    auc_best               = sk_auc(fpr_best, tpr_best)
    prec_best, rec_best, _ = precision_recall_curve(bvl, bvs)
    pr_auc_best            = sk_auc(rec_best, prec_best)

    # ── Figura ROC multiepoch ─────────────────────────────────────────────────
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for i, h in enumerate(historial_curvas):
        ax_roc.plot(h['fpr'], h['tpr'], color=colores[i], lw=1.2, alpha=0.75,
                    label=f"Epoch {h['epoch']}  (AUC={h['roc_auc']:.3f})")
    ax_roc.plot(fpr_best, tpr_best, color='black', lw=2.5,
                label=f"Mejor modelo — epoch {best_epoch_num}  (AUC={auc_best:.3f})")
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
                label='Aleatorio (AUC=0.5)')
    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.05])
    ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    ax_roc.set_title('Evolución curva ROC por epoch\nClasificador Borde / Interior', fontsize=12)
    ax_roc.legend(loc='lower right', fontsize=8, ncol=2)
    ax_roc.grid(True, alpha=0.3)
    fig_roc.tight_layout()
    path_roc_ev = os.path.join(save_dir, 'curvas_ROC_por_epoch.png')
    fig_roc.savefig(path_roc_ev, dpi=150, bbox_inches='tight')
    plt.close(fig_roc)
    print(f"  Curvas ROC por epoch guardadas en: {path_roc_ev}")

    # ── Figura PR multiepoch ──────────────────────────────────────────────────
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for i, h in enumerate(historial_curvas):
        ax_pr.plot(h['rec'], h['prec'], color=colores[i], lw=1.2, alpha=0.75,
                   label=f"Epoch {h['epoch']}  (AUC-PR={h['pr_auc']:.3f})")
    ax_pr.plot(rec_best, prec_best, color='black', lw=2.5,
               label=f"Mejor modelo — epoch {best_epoch_num}  (AUC-PR={pr_auc_best:.3f})")
    baseline = bvl.mean()
    ax_pr.axhline(y=baseline, color='gray', lw=1, linestyle='--',
                  label=f'Aleatorio (P={baseline:.2f})')
    ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1.05])
    ax_pr.set_xlabel('Recall (Sensibilidad)', fontsize=11)
    ax_pr.set_ylabel('Precision', fontsize=11)
    ax_pr.set_title('Evolución curva PR por epoch\nClasificador Borde / Interior', fontsize=12)
    ax_pr.legend(loc='upper right', fontsize=8, ncol=2)
    ax_pr.grid(True, alpha=0.3)
    fig_pr.tight_layout()
    path_pr_ev = os.path.join(save_dir, 'curvas_PR_por_epoch.png')
    fig_pr.savefig(path_pr_ev, dpi=150, bbox_inches='tight')
    plt.close(fig_pr)
    print(f"  Curvas PR  por epoch guardadas en: {path_pr_ev}")


def guardar_datos_excel(train_losses, val_losses,
                         historial_curvas, best_epoch_num,
                         best_val_scores, best_val_labels,
                         train_accs=None, val_accs=None,
                         test_scores=None, test_labels=None,
                         save_dir=None, model_path=None):
    """
    Guarda dos archivos con todos los datos de las curvas del entrenamiento.
    Puede llamarse dos veces: primero sin test (al acabar el entrenamiento)
    y luego con test (después de evaluar_test). En ambos casos las hojas de
    validación se escriben siempre, garantizando que nunca se pierdan.

    Excel (datos_curvas_entrenamiento.xlsx):
      - Perdidas_y_accuracy    : Epoch | Train_Loss | Val_Loss | Train_Acc | Val_Acc
      - Curva_ROC_val          : FPR | TPR  (mejor modelo validación)
      - Curva_PR_val           : Recall | Precision  (mejor modelo validación)
      - Curvas_ROC_por_epoch   : pares FPR/TPR por epoch + mejor al final
      - Curvas_PR_por_epoch    : pares Recall/Precision por epoch + mejor al final
      - Curva_umbral_val       : Umbral | Recall | Precision | F1
      - Curva_ROC_test         : FPR | TPR  (test, si se proporcionan)
      - Curva_PR_test          : Recall | Precision  (test, si se proporcionan)
      - Curva_umbral_test      : Umbral | Recall | Precision | F1  (test)

    CSV (metadata_entrenamiento.csv):
      - mejor_modelo_val : AUCs, métricas completas del mejor modelo de validación
      - checkpoint       : AUC-ROC y AUC-PR de cada checkpoint
      - test             : AUCs y métricas completas del set de test (si se proporciona)

    Args:
        train_losses     (list): pérdidas de entrenamiento por epoch.
        val_losses       (list): pérdidas de validación por epoch.
        historial_curvas (list): checkpoints ROC/PR por epoch.
        best_epoch_num   (int) : epoch del mejor modelo.
        best_val_scores  (list): scores del mejor modelo (validación).
        best_val_labels  (list): etiquetas del mejor modelo (validación).
        train_accs       (list|None): accuracy de entrenamiento por epoch.
        val_accs         (list|None): accuracy de validación por epoch.
        test_scores      (list|None): scores del set de test.
        test_labels      (list|None): etiquetas del set de test.
        save_dir         (str|None): directorio de guardado.
    """
    import os
    import pandas as pd
    from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve

    if save_dir is None:
        save_dir = "resultados_entrenamiento"
    os.makedirs(save_dir, exist_ok=True)
    path_excel = os.path.join(save_dir, 'datos_curvas_entrenamiento.xlsx')
    path_csv   = os.path.join(save_dir, 'metadata_entrenamiento.csv')

    if best_val_scores is None or best_val_labels is None:
        logger.warning("guardar_datos_excel: best_val_scores es None, no se puede generar el Excel.")
        print("  [AVISO] No hay scores de validación disponibles, se omite el guardado del Excel.")
        return

    bvs = np.array(best_val_scores)
    bvl = np.array(best_val_labels)

    fpr_b, tpr_b, _  = roc_curve(bvl, bvs)
    prec_b, rec_b, _ = precision_recall_curve(bvl, bvs)
    roc_auc_mejor    = sk_auc(fpr_b, tpr_b)
    pr_auc_mejor     = sk_auc(rec_b, prec_b)

    # Curva umbral del mejor modelo
    prec_th, rec_th, thresh_th = precision_recall_curve(bvl, bvs)
    prec_th = prec_th[:-1]; rec_th = rec_th[:-1]
    f1_th   = np.where((prec_th + rec_th) > 0,
                       2 * prec_th * rec_th / (prec_th + rec_th), 0)

    # Métricas al umbral óptimo del mejor modelo
    m = calcular_metricas(bvs, bvl)

    with pd.ExcelWriter(path_excel, engine='openpyxl') as writer:

        # Hoja 1: pérdidas y accuracy — siempre presente
        df_perd = {'Epoch': list(range(1, len(train_losses) + 1)),
                   'Train_Loss': train_losses, 'Val_Loss': val_losses}
        if train_accs is not None:
            df_perd['Train_Acc'] = train_accs
        if val_accs is not None:
            df_perd['Val_Acc'] = val_accs
        pd.DataFrame(df_perd).to_excel(writer, sheet_name='Perdidas_y_accuracy', index=False)

        # Hojas 2-3: ROC y PR validación — siempre presentes
        pd.DataFrame({'FPR': fpr_b, 'TPR': tpr_b}
                     ).to_excel(writer, sheet_name='Curva_ROC_val', index=False)
        pd.DataFrame({'Recall': rec_b, 'Precision': prec_b}
                     ).to_excel(writer, sheet_name='Curva_PR_val', index=False)

        # Hoja 4: ROC por epoch — pares FPR/TPR + mejor al final
        if historial_curvas:
            cols_roc = {}
            for h in historial_curvas:
                cols_roc[f"FPR_ep{h['epoch']}"] = pd.Series(h['fpr'])
                cols_roc[f"TPR_ep{h['epoch']}"] = pd.Series(h['tpr'])
            cols_roc['FPR_mejor'] = pd.Series(fpr_b)
            cols_roc['TPR_mejor'] = pd.Series(tpr_b)
            pd.concat(cols_roc, axis=1).to_excel(
                writer, sheet_name='Curvas_ROC_por_epoch', index=False)

        # Hoja 5: PR por epoch — pares Recall/Precision + mejor al final
        if historial_curvas:
            cols_pr = {}
            for h in historial_curvas:
                cols_pr[f"Recall_ep{h['epoch']}"] = pd.Series(h['rec'])
                cols_pr[f"Prec_ep{h['epoch']}"]   = pd.Series(h['prec'])
            cols_pr['Recall_mejor'] = pd.Series(rec_b)
            cols_pr['Prec_mejor']   = pd.Series(prec_b)
            pd.concat(cols_pr, axis=1).to_excel(
                writer, sheet_name='Curvas_PR_por_epoch', index=False)

        # Hoja 6: curva umbral validación — siempre presente
        pd.DataFrame({'Umbral': thresh_th, 'Recall': rec_th,
                      'Precision': prec_th, 'F1': f1_th}
                     ).to_excel(writer, sheet_name='Curva_umbral_val', index=False)

        # Hojas 7-9: test — solo si se proporcionan
        if test_scores is not None and test_labels is not None:
            ts = np.array(test_scores)
            tl = np.array(test_labels)
            fpr_t, tpr_t, _              = roc_curve(tl, ts)
            prec_t, rec_t, _             = precision_recall_curve(tl, ts)
            prec_tth, rec_tth, thresh_tth = precision_recall_curve(tl, ts)
            prec_tth = prec_tth[:-1]; rec_tth = rec_tth[:-1]
            f1_tth   = np.where((prec_tth + rec_tth) > 0,
                                2 * prec_tth * rec_tth / (prec_tth + rec_tth), 0)
            pd.DataFrame({'FPR': fpr_t, 'TPR': tpr_t}
                         ).to_excel(writer, sheet_name='Curva_ROC_test', index=False)
            pd.DataFrame({'Recall': rec_t, 'Precision': prec_t}
                         ).to_excel(writer, sheet_name='Curva_PR_test', index=False)
            pd.DataFrame({'Umbral': thresh_tth, 'Recall': rec_tth,
                          'Precision': prec_tth, 'F1': f1_tth}
                         ).to_excel(writer, sheet_name='Curva_umbral_test', index=False)

    print(f"  Datos de curvas guardados en: {path_excel}")
    logger.info(f"Datos Excel guardados en: {path_excel}")

    # ── CSV de metadata ───────────────────────────────────────────────────────
    filas_meta = []

    # Fila del mejor modelo de validación con métricas completas
    filas_meta.append({
        'tipo':          'mejor_modelo_val',
        'epoch':         best_epoch_num,
        'auc_roc':       round(roc_auc_mejor, 6),
        'auc_pr':        round(pr_auc_mejor,  6),
        'umbral_optimo': round(m['umbral_optimo'], 6),
        'TP':            m['TP'], 'TN': m['TN'], 'FP': m['FP'], 'FN': m['FN'],
        'recall':        round(m['recall'],    6),
        'precision':     round(m['precision'], 6),
        'f1':            round(m['f1'],        6),
        'kappa':         round(m['kappa'],     6),
    })

    # Una fila por checkpoint con solo AUC
    for h in historial_curvas:
        filas_meta.append({
            'tipo': 'checkpoint', 'epoch': h['epoch'],
            'auc_roc': round(h['roc_auc'], 6), 'auc_pr': round(h['pr_auc'], 6),
            'umbral_optimo': '', 'TP': '', 'TN': '', 'FP': '', 'FN': '',
            'recall': '', 'precision': '', 'f1': '', 'kappa': '',
        })

    # Fila de test con métricas completas — si se proporcionan
    if test_scores is not None and test_labels is not None:
        ts = np.array(test_scores); tl = np.array(test_labels)
        fpr_t, tpr_t, _  = roc_curve(tl, ts)
        prec_t, rec_t, _ = precision_recall_curve(tl, ts)
        # Cargar umbral del modelo para calcular métricas de test con el mismo
        # umbral que usó la red, no con el que maximiza F1 sobre el test
        _umbral_test = None
        if model_path is not None:
            import os as _os
            _umbral_path = model_path.replace('.pth', '_umbral_optimo.txt')
            if _os.path.exists(_umbral_path):
                with open(_umbral_path) as _f:
                    _umbral_test = float(_f.read().strip())
        if _umbral_test is not None:
            m_test = calcular_metricas_con_umbral(ts, tl, _umbral_test)
        else:
            logger.warning("guardar_datos_excel: umbral del modelo no encontrado, "
                           "usando umbral óptimo del test como fallback.")
            m_test = calcular_metricas(ts, tl)
        filas_meta.append({
            'tipo':          'test',
            'epoch':         '',
            'auc_roc':       round(sk_auc(fpr_t, tpr_t), 6),
            'auc_pr':        round(sk_auc(rec_t, prec_t), 6),
            'umbral_optimo': round(m_test['umbral_optimo'], 6),
            'TP':            m_test['TP'], 'TN': m_test['TN'],
            'FP':            m_test['FP'], 'FN': m_test['FN'],
            'recall':        round(m_test['recall'],    6),
            'precision':     round(m_test['precision'], 6),
            'f1':            round(m_test['f1'],        6),
            'kappa':         round(m_test['kappa'],     6),
        })

    pd.DataFrame(filas_meta).to_csv(path_csv, index=False)
    print(f"  Metadata guardada en:         {path_csv}")
    logger.info(f"Metadata CSV guardada en: {path_csv}")


def bootstrap_auc(scores, labels, n_iter=1000, ci=95, seed=42):
    """
    Calcula el intervalo de confianza del AUC-ROC y AUC-PR mediante bootstrap.

    El bootstrap funciona así: dado un conjunto de N predicciones reales,
    se remuestrea con reemplazamiento N veces (puede salir la misma partícula
    varias veces y otras no salir). Se calcula el AUC sobre ese remuestreo.
    Repitiendo esto n_iter veces se obtiene una distribución de AUCs posibles.
    Los percentiles (100-ci)/2 y (100+ci)/2 de esa distribución son los límites
    del intervalo de confianza. Con 1000 iteraciones y ci=95 se toman los
    percentiles 2.5 y 97.5.

    Args:
        scores (np.ndarray): Probabilidades de clase Interior.
        labels (np.ndarray): Etiquetas reales (0=Borde, 1=Interior).
        n_iter (int):        Número de remuestreos (default: 1000).
        ci     (int):        Nivel de confianza en % (default: 95).
        seed   (int):        Semilla para reproducibilidad (default: 42).

    Returns:
        dict con claves:
            roc_auc_mean, roc_auc_lo, roc_auc_hi,
            pr_auc_mean,  pr_auc_lo,  pr_auc_hi
    """
    from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve

    rng = np.random.default_rng(seed)
    n   = len(scores)
    roc_aucs = []
    pr_aucs  = []

    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)   # remuestreo con reemplazamiento
        s_b = scores[idx]
        l_b = labels[idx]
        # Saltar si el remuestreo solo tiene una clase (AUC indefinido)
        if len(np.unique(l_b)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(l_b, s_b)
        roc_aucs.append(sk_auc(fpr_b, tpr_b))
        prec_b, rec_b, _ = precision_recall_curve(l_b, s_b)
        pr_aucs.append(sk_auc(rec_b, prec_b))

    alpha = (100 - ci) / 2
    return {
        'roc_auc_mean': float(np.mean(roc_aucs)),
        'roc_auc_lo':   float(np.percentile(roc_aucs, alpha)),
        'roc_auc_hi':   float(np.percentile(roc_aucs, 100 - alpha)),
        'pr_auc_mean':  float(np.mean(pr_aucs)),
        'pr_auc_lo':    float(np.percentile(pr_aucs, alpha)),
        'pr_auc_hi':    float(np.percentile(pr_aucs, 100 - alpha)),
    }


def evaluar_test(model_path, tif_path, csv_path, patch_size=64,
                 save_dir=None, n_bootstrap=1000):
    """
    Modo test: carga el modelo entrenado y evalúa sobre un set de test etiquetado.
    Genera las mismas curvas que en validación (ROC, PR, umbral) con prefijo 'test',
    calcula intervalos de confianza mediante bootstrap, imprime todo por consola
    y devuelve scores y etiquetas para el Excel.

    Los datos de test deben estar en CSVs con columna 'clase', igual que los de
    entrenamiento. Las partículas 'aislada' se excluyen automáticamente.

    Usa el mismo camino de extracción de patches que _generar_csvs_test
    (partícula a partícula con extract_patch_around_particle), garantizando que
    los scores aquí calculados y los del CSV clasificado son idénticos.

    Args:
        model_path   (str):           Ruta al .pth del modelo entrenado.
        tif_path     (str|list[str]): Ruta/s al .tif de test.
        csv_path     (str|list[str]): Ruta/s al CSV de test (con columna 'clase').
        patch_size   (int):           Tamaño del recorte (default: 64).
        save_dir     (str|None):      Directorio donde guardar las curvas de test.
        n_bootstrap  (int):           Iteraciones bootstrap para IC 95% (default: 1000).

    Returns:
        tuple: (test_scores, test_labels) — arrays numpy listos para el Excel.
    """
    import os
    import pandas as pd
    from funciones_auxiliares import load_tif_image, extract_patch_around_particle

    if isinstance(tif_path, str):
        tif_path = [tif_path]
    if isinstance(csv_path, str):
        csv_path = [csv_path]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    model = MitochondriaContextCNN(num_channels=2, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ── Cargar umbral óptimo guardado en entrenamiento ────────────────────────
    umbral_path = model_path.replace('.pth', '_umbral_optimo.txt')
    if os.path.exists(umbral_path):
        with open(umbral_path) as f:
            umbral = float(f.read().strip())
        print(f"  Umbral cargado para test: {umbral:.4f}")
    else:
        umbral = 0.5
        print(f"  [AVISO] Sin umbral guardado, usando 0.5 para test.")

    # ── Extraer scores partícula a partícula (igual que _generar_csvs_test) ───
    # Mismo bucle que _generar_csvs_test para garantizar scores idénticos
    print(f"\n  Cargando datos de test...")
    test_scores = []
    test_labels = []

    CLASS_MAP_TEST = {'borde': 0, 'interior': 1, 'exterior': 1}

    for tif, csv in zip(tif_path, csv_path):
        df = pd.read_csv(csv, encoding='utf-8-sig')
        cols_lower = {c.strip().lower(): c for c in df.columns}
        col_y     = cols_lower.get('y', 'y')
        col_x     = cols_lower.get('x', 'x')
        col_clase = cols_lower.get('clase', None)

        if col_clase is None:
            raise ValueError(f"El CSV de test '{csv}' no tiene columna 'clase'.")

        clases_raw    = df[col_clase].str.strip().str.lower().tolist()
        mask_aisladas = np.array([c == 'aislada' for c in clases_raw])

        canal_rojo, canal_verde = load_tif_image(tif)

        with torch.no_grad():
            for i, row in df.iterrows():
                if mask_aisladas[i]:
                    continue  # aisladas excluidas
                label = CLASS_MAP_TEST.get(clases_raw[i])
                if label is None:
                    continue  # etiqueta desconocida — saltar
                pos   = (float(row[col_y]), float(row[col_x]))
                patch = extract_patch_around_particle(
                    canal_rojo, canal_verde, pos, patch_size)
                patch_t = torch.FloatTensor(patch).unsqueeze(0).to(device)
                output  = model(patch_t)
                prob    = torch.softmax(output, dim=1)[0, 1].item()
                test_scores.append(prob)
                test_labels.append(label)

    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)

    print(f"  Test: {len(test_labels)} partículas evaluadas "
          f"(borde={int((test_labels==0).sum())}, "
          f"interior={int((test_labels==1).sum())})")
    logger.info(f"Test: {len(test_labels)} partículas evaluadas")

    # ── Curvas y métricas de test ─────────────────────────────────────────────
    plot_curvas_evaluacion(test_scores, test_labels,
                           save_dir=save_dir, prefijo='test',
                           umbral_externo=umbral)

    # ── Intervalos de confianza bootstrap ─────────────────────────────────────
    print(f"\n  Calculando intervalos de confianza (bootstrap, {n_bootstrap} iter.)...")
    ic = bootstrap_auc(test_scores, test_labels, n_iter=n_bootstrap)
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  INTERVALOS DE CONFIANZA 95% — TEST (bootstrap)")
    print(f"{sep}")
    print(f"  AUC-ROC : {ic['roc_auc_mean']:.4f}  "
          f"(IC 95%: {ic['roc_auc_lo']:.4f} – {ic['roc_auc_hi']:.4f})")
    print(f"  AUC-PR  : {ic['pr_auc_mean']:.4f}  "
          f"(IC 95%: {ic['pr_auc_lo']:.4f} – {ic['pr_auc_hi']:.4f})")
    print(f"{sep}\n")
    logger.info(f"Bootstrap test — AUC-ROC: {ic['roc_auc_mean']:.4f} "
                f"[{ic['roc_auc_lo']:.4f}, {ic['roc_auc_hi']:.4f}] | "
                f"AUC-PR: {ic['pr_auc_mean']:.4f} "
                f"[{ic['pr_auc_lo']:.4f}, {ic['pr_auc_hi']:.4f}]")

    # ── Generar CSV clasificado por archivo de test ───────────────────────────
    _generar_csvs_test(tif_path, csv_path, model, patch_size,
                       umbral, save_dir)

    return test_scores, test_labels


def _generar_csvs_test(tif_path, csv_path, model, patch_size,
                        umbral, save_dir):
    """
    Genera un CSV clasificado por cada archivo de test, equivalente al que
    produce save_results_to_csv en clasificacion.py. Se guarda en save_dir
    con el nombre <nombre_original>_clasificado_test.csv.

    Columnas añadidas al CSV original:
      - clasificacion  : 'Borde', 'Interior' o 'Aislada'
      - prob_borde     : probabilidad de clase Borde [0,1]    (índice 0)
      - prob_interior  : probabilidad de clase Interior [0,1] (índice 1)

    Args:
        tif_path   (str | list[str]): Ruta/s al .tif de test.
        csv_path   (str | list[str]): Ruta/s al CSV de test (con columna 'clase').
        model      (nn.Module):       Modelo ya cargado y en eval() desde evaluar_test.
        patch_size (int):             Tamaño del recorte.
        umbral     (float):           Umbral de decisión cargado del entrenamiento.
        save_dir   (str | None):      Directorio de guardado.
    """
    import os
    import pandas as pd

    if isinstance(tif_path, str):
        tif_path = [tif_path]
    if isinstance(csv_path, str):
        csv_path = [csv_path]

    if save_dir is None:
        save_dir = "resultados_entrenamiento"
    os.makedirs(save_dir, exist_ok=True)

    device = next(model.parameters()).device

    CLASS_NAMES = ['Borde', 'Interior']

    for tif, csv in zip(tif_path, csv_path):
        # ── Leer CSV original ────────────────────────────────────────────────
        df = pd.read_csv(csv, encoding='utf-8-sig')
        cols_lower = {c.strip().lower(): c for c in df.columns}

        col_y = cols_lower.get('y', 'y')
        col_x = cols_lower.get('x', 'x')
        col_clase = cols_lower.get('clase', None)

        # ── Determinar máscara de aisladas ───────────────────────────────────
        if col_clase is not None:
            clases_raw = df[col_clase].str.strip().str.lower().tolist()
            mask_aisladas = np.array([c == 'aislada' for c in clases_raw])
        else:
            mask_aisladas = np.zeros(len(df), dtype=bool)

        # ── Cargar imagen y clasificar partículas no aisladas ─────────────────
        from funciones_auxiliares import load_tif_image, extract_patch_around_particle
        canal_rojo, canal_verde = load_tif_image(tif)

        clasificacion = []
        prob_borde    = []
        prob_interior = []

        with torch.no_grad():
            for i, row in df.iterrows():
                if mask_aisladas[i]:
                    clasificacion.append('Aislada')
                    prob_borde.append(np.nan)
                    prob_interior.append(np.nan)
                else:
                    pos = (float(row[col_y]), float(row[col_x]))
                    patch = extract_patch_around_particle(
                        canal_rojo, canal_verde, pos, patch_size)
                    patch_t = torch.FloatTensor(patch).unsqueeze(0).to(device)
                    output = model(patch_t)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    pred  = 1 if probs[1] >= umbral else 0
                    clasificacion.append(CLASS_NAMES[pred])
                    prob_borde.append(round(float(probs[0]), 4))
                    prob_interior.append(round(float(probs[1]), 4))

        df['clasificacion'] = clasificacion
        df['prob_borde']    = prob_borde
        df['prob_interior'] = prob_interior

        # ── Guardar en resultados_entrenamiento ──────────────────────────────
        nombre_base = os.path.splitext(os.path.basename(csv))[0]
        out_name    = f"{nombre_base}_clasificado_test.csv"
        out_path    = os.path.join(save_dir, out_name)
        df.to_csv(out_path, index=False)
        print(f"  CSV clasificado test guardado en: {out_path}")
        logger.info(f"CSV clasificado test guardado en: {out_path}")


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001,
                save_dir=None, model_path='best_mito_classifier.pth'):
    '''
    Ejecuta el ciclo completo de entrenamiento y validación.

    Criterio de guardado: el modelo se sobreescribe en disco únicamente si
    el AUC-ROC de validación de la epoch actual supera el mejor AUC histórico
    registrado en <model_path reemplazando .pth por _best_auc.txt>.
    Esto garantiza que entrenamientos sucesivos nunca empeoran el modelo
    guardado, independientemente de cuántas veces se ejecute el script.

    Al finalizar, si se ha mejorado el modelo guardado, se generan y guardan
    (en save_dir) la curva ROC, la curva PR y la curva de umbral, y se
    imprimen por consola y en el log todas las métricas: Recall, Precision,
    F1, Cohen's Kappa y AUC.

    Args:
        model        (nn.Module):  MitochondriaContextCNN.
        train_loader (DataLoader): Datos de entrenamiento.
        val_loader   (DataLoader): Datos de validación.
        num_epochs   (int):        Número de epochs (default: 50).
        lr           (float):      Tasa de aprendizaje inicial (default: 0.001).
        save_dir     (str|None):   Directorio donde guardar las curvas.
        model_path   (str):        Ruta del archivo .pth del modelo.

    Returns:
        tuple: (modelo_entrenado, train_losses, val_losses)
    '''
    import os

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = model.to(device)

    criterion = nn.CrossEntropyLoss() #mirar si cambiamos por la binaria
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # ── Cargar el mejor AUC histórico guardado en disco ───────────────────────
    # Se guarda en un archivo de texto auxiliar junto al .pth para que
    # entrenamientos sucesivos nunca sobreescriban un modelo mejor.
    auc_record_path = model_path.replace('.pth', '_best_auc.txt')
    if os.path.exists(auc_record_path):
        with open(auc_record_path) as f:
            best_val_auc = float(f.read().strip())
        logger.info(f"AUC histórico cargado desde disco: {best_val_auc:.4f}. "
                    f"Solo se guardará si este entrenamiento lo supera.")
        print(f"  AUC histórico en disco: {best_val_auc:.4f} — "
              f"el modelo solo se actualizará si se supera.")
    else:
        best_val_auc = 0.0
        logger.info("No hay AUC histórico guardado. Se guardará el mejor de este entrenamiento.")

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []   # accuracy por epoch
    best_val_scores = None
    best_val_labels = None
    best_epoch_auc  = 0.0   # mejor AUC de este entrenamiento (independiente del histórico)
    modelo_guardado = False  # flag explícito para evitar comparación de floats al final
    best_epoch_num  = 0      # número de epoch del mejor modelo guardado
    historial_curvas = []    # checkpoints ROC/PR cada 5 epochs a partir de la 30
    epoch_checkpoint_inicio = 30  # a partir de qué epoch guardar checkpoints
    epoch_checkpoint_cada   = 5   # cada cuántas epochs guardar un checkpoint

    logger.info(f"Iniciando entrenamiento: {num_epochs} epochs, LR: {lr}")

    for epoch in range(num_epochs):
        model.train()
        train_loss    = 0
        train_correct = 0
        train_total   = 0

        for patches, labels in train_loader:
            patches = patches.to(device)
            labels  = labels.squeeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(patches)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        model.eval()
        val_loss      = 0
        val_correct   = 0
        val_total     = 0
        val_scores_ep = [] #lista que contendrá las probabilidades asociadas a la clasificacion INTERIOR
        val_labels_ep = [] #lista que contendrá la etiqueta real (la que viene en el csv)

        with torch.no_grad():
            for patches, labels in val_loader:
                patches = patches.to(device)
                labels  = labels.squeeze(1).to(device)
                outputs  = model(patches)
                loss     = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() #me coge solo las probabilidades interior (columna 1)
                val_scores_ep.extend(probs)
                val_labels_ep.extend(labels.cpu().numpy())
                preds       = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss     / len(val_loader))
        train_accs.append(100 * train_correct / train_total)
        val_accs.append(100   * val_correct   / val_total)
        scheduler.step(val_losses[-1])

        fpr, tpr, _ = roc_curve(val_labels_ep, val_scores_ep)
        #barre los valores obtenidos probando distintos umbrales,ie, dice pues a partir de 0.9 es positivo y así va calculando puntos fpr y tpr necesarios para representar la curva.
        val_auc      = auc(fpr, tpr) #funcion que calcula el auc.

        logger.info(f'Epoch {epoch+1:2d}: Val AUC {val_auc:.4f} | '
                    f'Train Loss {train_losses[-1]:.3f} | Val Loss {val_losses[-1]:.3f}')

        # Actualizar mejor epoch de este entrenamiento (siempre, independiente del histórico)
        if val_auc > best_epoch_auc:
            best_epoch_auc  = val_auc
            best_val_scores = val_scores_ep
            best_val_labels = val_labels_ep

        if val_auc > best_val_auc:
            best_val_auc    = val_auc
            modelo_guardado = True  # marcar que se ha guardado en disco en este entrenamiento
            best_epoch_num  = epoch + 1  # guardar el número de epoch del mejor modelo

            # Calcular y guardar el umbral óptimo junto al modelo
            m_epoch     = calcular_metricas(val_scores_ep, val_labels_ep)
            umbral_path = model_path.replace('.pth', '_umbral_optimo.txt')

            torch.save(model.state_dict(), model_path)
            with open(auc_record_path, 'w') as f:
                f.write(f"{best_val_auc:.6f}")
            with open(umbral_path, 'w') as f:
                f.write(f"{m_epoch['umbral_optimo']:.6f}")

            print(f'  -> Modelo guardado (val_auc={val_auc:.4f}, '
                  f'umbral_optimo={m_epoch["umbral_optimo"]:.4f})')

        # ── Checkpoint ROC/PR cada 5 epochs a partir de la epoch 30 ──────────
        ep_num = epoch + 1
        if ep_num >= epoch_checkpoint_inicio and ep_num % epoch_checkpoint_cada == 0:
            from sklearn.metrics import precision_recall_curve, auc as sk_auc2
            prec_cp, rec_cp, _ = precision_recall_curve(val_labels_ep, val_scores_ep)
            historial_curvas.append({
                'epoch':   ep_num,
                'fpr':     fpr,
                'tpr':     tpr,
                'roc_auc': val_auc,
                'prec':    prec_cp,
                'rec':     rec_cp,
                'pr_auc':  sk_auc2(rec_cp, prec_cp),
            })

    # ── Curvas y métricas: siempre con el mejor epoch de este entrenamiento ──
    print(f'\nEntrenamiento completado.')
    print(f'  Mejor AUC de este entrenamiento : {best_epoch_auc:.4f}')
    print(f'  Mejor AUC histórico en disco    : {best_val_auc:.4f}')
    if modelo_guardado:
        print(f'  -> El modelo en disco ha sido actualizado.')
        plot_curvas_evaluacion(best_val_scores, best_val_labels, save_dir=save_dir, prefijo='val')
    else:
        print(f'  -> El modelo en disco NO se ha actualizado (AUC histórico superior).')
        print(f'     Se muestran las curvas del mejor epoch de este entrenamiento.')
        plot_curvas_evaluacion(best_val_scores, best_val_labels, save_dir=None, prefijo='val')
    plot_training_curves(train_losses, val_losses, train_accs=train_accs, val_accs=val_accs, save_dir=save_dir)

    # ── Curvas ROC y PR multiepoch (checkpoints cada 5 epochs desde la 30) ───
    _save_dir_efectivo = save_dir if modelo_guardado else None
    plot_curvas_por_epoch(historial_curvas, best_epoch_num,
                          best_val_scores, best_val_labels,
                          save_dir=_save_dir_efectivo)

    # ── Exportar todos los datos de curvas a Excel + metadata a CSV ──────────
    # Se guarda siempre (independientemente de si se actualizó el modelo en disco)
    # usando los scores del mejor epoch de este entrenamiento
    guardar_datos_excel(train_losses, val_losses,
                        historial_curvas, best_epoch_num,
                        best_val_scores, best_val_labels,
                        train_accs=train_accs, val_accs=val_accs,
                        save_dir=save_dir)

    return (model, train_losses, val_losses,
            train_accs, val_accs,
            historial_curvas, best_epoch_num,
            best_val_scores, best_val_labels)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Opción A: una sola imagen ─────────────────────────────────────────────
    # TIF_PATH = "datos/SUb_01_2_merged.tif"
    # CSV_PATH = "datos/SUb_01_2_datos_training.csv"

    # ── Opción B: listas manuales ─────────────────────────────────────────────
    # TIF_PATH = ["datos/SUb_01_2_merged.tif", "datos/SUb_01_5_merged.tif"]
    # CSV_PATH = ["datos/SUb_01_2_datos.csv",  "datos/SUb_01_5_datos.csv"]

    # ── Opción C: directorio completo (útil para datos aumentados) ────────────
    TIF_PATH, CSV_PATH = load_pairs_from_dir("../data_augmentation_clase")

    # Directorio donde se guardarán todas las curvas, el Excel y el CSV de metadata
    # Ponlo a None si no quieres guardarlas en disco
    SAVE_DIR = "resultados_entrenamiento"

    train_loader, val_loader = prepare_loaders(TIF_PATH, CSV_PATH)

    # ── Visualización opcional de patches ─────────────────────────────────────
    #visualize_loader(train_loader, n_patches=20, save_dir="vis_patches3/", save_tiff=False)

    MODEL_PATH = 'best_mito_classifier.pth'

    model = MitochondriaContextCNN(num_channels=2, num_classes=2)
    (model, train_losses, val_losses,
     train_accs, val_accs,
     historial_curvas, best_epoch_num,
     best_val_scores, best_val_labels) = train_model(
        model, train_loader, val_loader, num_epochs=300, lr=0.001,
        save_dir=SAVE_DIR, model_path=MODEL_PATH)

    # ── Modo test (descomentar cuando tengas el set de test listo) ────────
    TEST_TIF, TEST_CSV = load_pairs_from_dir('../datos_test')
    test_scores, test_labels = evaluar_test(
         model_path = MODEL_PATH,
         tif_path   = TEST_TIF,
         csv_path   = TEST_CSV,
         save_dir   = SAVE_DIR)
    # Actualizar el Excel con los datos de test
    guardar_datos_excel(
         train_losses, val_losses,
         historial_curvas, best_epoch_num,
         best_val_scores, best_val_labels,
         train_accs=train_accs, val_accs=val_accs,
         test_scores=test_scores, test_labels=test_labels,
         save_dir=SAVE_DIR, model_path=MODEL_PATH)