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
        val_scores (list of float): Probabilidades de clase Borde (clase positiva).
        val_labels (list of int):   Etiquetas reales (0=No borde, 1=Borde).

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
    idx_opt      = int(np.argmax(f1s))
    umbral_optimo = float(thresholds[idx_opt])

    # Calcular TP, TN, FP, FN con ese umbral
    preds = (val_scores >= umbral_optimo).astype(int)
    TP = int(((preds == 1) & (val_labels == 1)).sum())
    TN = int(((preds == 0) & (val_labels == 0)).sum())
    FP = int(((preds == 1) & (val_labels == 0)).sum())
    FN = int(((preds == 0) & (val_labels == 1)).sum())

    recall    = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0.0
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


def plot_curvas_evaluacion(val_scores, val_labels, save_dir=None):
    '''
    Calcula todas las métricas de evaluación del mejor modelo, las imprime
    por consola y las registra en el log. Genera dos figuras independientes:
      - Curva ROC (FPR vs TPR) con AUC en la leyenda.
      - Curva PR  (Recall vs Precision) con Average Precision en la leyenda.

    Las figuras se guardan siempre en save_dir (si se indica) y además se
    intentan abrir automáticamente con el visor de imágenes del sistema.
    Esto funciona tanto en local (abre ventana) como en servidor remoto
    accedido por SSH con -X (reenvío X11).

    El criterio de guardado del modelo sigue siendo exclusivamente el AUC-ROC.
    Las métricas adicionales (Recall, Precision, F1, Kappa) se calculan al
    umbral óptimo que maximiza F1, solo a efectos informativos.
    Las métricas NO se muestran dentro de las gráficas — solo por consola y log.

    Args:
        val_scores (list of float): Probabilidades de clase Borde (validación).
        val_labels (list of int):   Etiquetas reales (0=No borde, 1=Borde).
        save_dir   (str | None):    Directorio donde guardar las dos figuras.
                   Si es None se usa un directorio temporal.
                   Se crean: <save_dir>/curva_ROC.png y <save_dir>/curva_PR.png

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
    avg_precision          = average_precision_score(val_labels, val_scores)

    # ── Métricas al umbral óptimo (max F1) ───────────────────────────────────
    m = calcular_metricas(val_scores, val_labels)

    # ── Consola ───────────────────────────────────────────────────────────────
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  MÉTRICAS DE VALIDACIÓN — MEJOR MODELO")
    print(f"{sep}")
    print(f"  Criterio de guardado   : AUC-ROC = {roc_auc:.4f}")
    print(f"  Umbral óptimo (max F1) : {m['umbral_optimo']:.4f}")
    print(f"  TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")
    print(f"  {'─'*46}")
    print(f"  Recall         : {m['recall']:6.2f} %")
    print(f"  Precision      : {m['precision']:6.2f} %")
    print(f"  F1 score       : {m['f1']:6.2f} %")
    print(f"  Cohen's Kappa  : {m['kappa']:6.4f}")
    print(f"  AUC-ROC        : {roc_auc:6.4f}")
    print(f"  Avg. Precision : {avg_precision:6.4f}  (área bajo curva PR)")
    print(f"{sep}\n")

    # ── Log ───────────────────────────────────────────────────────────────────
    logger.info("── Métricas del mejor modelo (validación) ──")
    logger.info(f"  AUC-ROC        : {roc_auc:.4f}  (criterio de guardado)")
    logger.info(f"  Umbral opt.    : {m['umbral_optimo']:.4f}  (maximiza F1)")
    logger.info(f"  TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")
    logger.info(f"  Recall         : {m['recall']:.2f} %")
    logger.info(f"  Precision      : {m['precision']:.2f} %")
    logger.info(f"  F1 score       : {m['f1']:.2f} %")
    logger.info(f"  Cohen's Kappa  : {m['kappa']:.4f}")
    logger.info(f"  Avg. Precision : {avg_precision:.4f}")

    # ── Rutas de guardado ─────────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = "resultados_entrenamiento"
        os.makedirs(save_dir, exist_ok=True)
    path_roc = os.path.join(save_dir, "curva_ROC.png")
    path_pr  = os.path.join(save_dir, "curva_PR.png")

    # ── Curva ROC ─────────────────────────────────────────────────────────────
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    ax_roc.plot(fpr, tpr, color='steelblue', lw=2,
                label=f'Curva ROC  (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=1.2,
                linestyle='--', label='Clasificador aleatorio (AUC = 0.5)')
    ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR / Recall)', fontsize=12)
    ax_roc.set_title('Curva ROC — Validación del mejor modelo\n'
                     'Clasificador Borde / No borde', fontsize=13)
    ax_roc.legend(loc='lower right', fontsize=11)
    ax_roc.grid(True, alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(path_roc, dpi=150, bbox_inches='tight')
    plt.close(fig_roc)
    print(f"  Curva ROC guardada en: {path_roc}")

    # ── Curva PR ──────────────────────────────────────────────────────────────
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    ax_pr.plot(recalls, precisions, color='darkorange', lw=2,
               label=f'Curva PR  (AP = {avg_precision:.3f})')
    baseline = val_labels.mean()
    ax_pr.axhline(y=baseline, color='gray', lw=1.2, linestyle='--',
                  label=f'Clasificador aleatorio (P = {baseline:.2f})')
    ax_pr.set_xlim([0.0, 1.0]); ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall (Sensibilidad)', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Curva PR — Validación del mejor modelo\n'
                    'Clasificador Borde / No borde', fontsize=13)
    ax_pr.legend(loc='upper right', fontsize=11)
    ax_pr.grid(True, alpha=0.3)
    fig_pr.tight_layout()
    fig_pr.savefig(path_pr, dpi=150, bbox_inches='tight')
    plt.close(fig_pr)
    print(f"  Curva PR  guardada en: {path_pr}")

    return roc_auc


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
    (en save_dir) la curva ROC y la curva PR, y se imprimen por consola y en
    el log todas las métricas: Recall, Precision, F1, Cohen's Kappa y AUC.

    Args:
        model        (nn.Module):  MitochondriaContextCNN.
        train_loader (DataLoader): Datos de entrenamiento.
        val_loader   (DataLoader): Datos de validación.
        num_epochs   (int):        Número de epochs (default: 50).
        lr           (float):      Tasa de aprendizaje inicial (default: 0.001).
        save_dir     (str|None):   Directorio donde guardar curva_ROC.png y
                                   curva_PR.png. None = no guardar en disco.
        model_path   (str):        Ruta del archivo .pth del modelo.

    Returns:
        tuple: (modelo_entrenado, train_losses, val_losses)
    '''
    import os

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = model.to(device)

    criterion = nn.CrossEntropyLoss()
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
    best_val_scores = None
    best_val_labels = None
    best_epoch_auc  = 0.0   # mejor AUC de este entrenamiento (independiente del histórico)

    logger.info(f"Iniciando entrenamiento: {num_epochs} epochs, LR: {lr}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for patches, labels in train_loader:
            patches = patches.to(device)
            labels  = labels.squeeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(patches)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss      = 0
        val_scores_ep = []
        val_labels_ep = []

        with torch.no_grad():
            for patches, labels in val_loader:
                patches = patches.to(device)
                labels  = labels.squeeze(1).to(device)
                outputs  = model(patches)
                loss     = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                val_scores_ep.extend(probs)
                val_labels_ep.extend(labels.cpu().numpy())

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss     / len(val_loader))
        scheduler.step(val_losses[-1])

        fpr, tpr, _ = roc_curve(val_labels_ep, val_scores_ep)
        val_auc      = auc(fpr, tpr)

        logger.info(f'Epoch {epoch+1:2d}: Val AUC {val_auc:.4f} | '
                    f'Train Loss {train_losses[-1]:.3f} | Val Loss {val_losses[-1]:.3f}')

        if val_auc > best_val_auc:
            best_val_auc    = val_auc
            best_val_scores = val_scores_ep
            best_val_labels = val_labels_ep

            # Calcular y guardar el umbral óptimo junto al modelo
            m_epoch          = calcular_metricas(val_scores_ep, val_labels_ep)
            umbral_path      = model_path.replace('.pth', '_umbral_optimo.txt')

            torch.save(model.state_dict(), model_path)
            with open(auc_record_path, 'w') as f:
                f.write(f"{best_val_auc:.6f}")
            with open(umbral_path, 'w') as f:
                f.write(f"{m_epoch['umbral_optimo']:.6f}")

            print(f'  -> Modelo guardado (val_auc={val_auc:.4f}, '
                  f'umbral_optimo={m_epoch["umbral_optimo"]:.4f})')
        elif val_auc > best_epoch_auc:
            # Mejor epoch de este entrenamiento aunque no supere el histórico:
            # guardamos los scores para plotear las curvas igualmente al final
            best_epoch_auc    = val_auc
            best_val_scores   = val_scores_ep
            best_val_labels   = val_labels_ep

    # ── Curvas y métricas: siempre con el mejor epoch de este entrenamiento ──
    modelo_guardado = best_epoch_auc == best_val_auc  # True si se mejoró el histórico
    print(f'\nEntrenamiento completado.')
    print(f'  Mejor AUC de este entrenamiento : {best_epoch_auc:.4f}')
    print(f'  Mejor AUC histórico en disco    : {best_val_auc:.4f}')
    if modelo_guardado:
        print(f'  -> El modelo en disco ha sido actualizado.')
        plot_curvas_evaluacion(best_val_scores, best_val_labels, save_dir=save_dir)
    else:
        print(f'  -> El modelo en disco NO se ha actualizado (AUC histórico superior).')
        print(f'     Se muestran las curvas del mejor epoch de este entrenamiento.')
        plot_curvas_evaluacion(best_val_scores, best_val_labels, save_dir=None)

    return model, train_losses, val_losses


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

    # Directorio donde se guardarán curva_ROC.png y curva_PR.png
    # Ponlo a None si no quieres guardarlas en disco
    SAVE_DIR = "resultados_entrenamiento"

    train_loader, val_loader = prepare_loaders(TIF_PATH, CSV_PATH)

    # ── Visualización opcional de batches ─────────────────────────────────────
    # visualize_loader(train_loader, n_batches=2, save_dir="vis_batches")

    model = MitochondriaContextCNN(num_channels=2, num_classes=2)
    train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001,
                save_dir=SAVE_DIR,
                model_path='best_mito_classifier.pth')