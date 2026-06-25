import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

from funciones_auxiliares import (load_tif_image,
                                   extract_patch_around_particle,
                                   detectar_aisladas_EDT,
                                   load_pairs_from_dir)
from modelo_CNN import MitochondriaContextCNN

CLASS_NAMES = ['Borde', 'Interior']


# ──────────────────────────────────────────────────────────────────────────────
# Clasificación de una sola partícula
# ──────────────────────────────────────────────────────────────────────────────

def classify_single_particle(model, canal_rojo, canal_verde, particle_center,
                              patch_size=64, device=None, umbral=0.5):
    """
    Clasifica UNA partícula pasando su recorte de 2 canales por la red.

    La predicción se obtiene comparando la probabilidad de clase Borde con
    el umbral indicado. Por defecto es 0.5, pero se recomienda usar el
    umbral óptimo guardado durante el entrenamiento (el que maximiza F1
    sobre validación), cargado automáticamente en classify_from_tif.

    Args:
        model           : Red neuronal entrenada.
        canal_rojo      (np.ndarray): Canal rojo normalizado (H, W).
        canal_verde     (np.ndarray): Canal verde normalizado (H, W).
        particle_center (tuple): Coordenadas (y, x).
        patch_size      (int): Tamaño del recorte (default: 64).
        device          : Dispositivo torch.
        umbral          (float): Umbral de decisión sobre prob. de clase Interior (índice 1).

    Returns:
        tuple: (prediction_idx, probabilities, class_name)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patch = extract_patch_around_particle(canal_rojo, canal_verde,
                                          particle_center, patch_size)
    patch_tensor = torch.FloatTensor(patch).unsqueeze(0).to(device)

    with torch.no_grad():
        output        = model(patch_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        # Usar el umbral óptimo en lugar del argmax implícito (umbral=0.5)
        prediction    = 1 if probabilities[1] >= umbral else 0

    return prediction, probabilities, CLASS_NAMES[prediction]


# ──────────────────────────────────────────────────────────────────────────────
# Clasificación sobre uno o varios TIFFs
# ──────────────────────────────────────────────────────────────────────────────

def classify_from_tif(model_path, tif_path, positions, patch_size=64):
    """
    Carga el modelo y clasifica listas de posiciones sobre uno o varios TIFFs.

    Carga automáticamente el umbral óptimo guardado durante el entrenamiento
    (fichero <model_path reemplazando .pth por _umbral_optimo.txt>). Si ese
    fichero no existe usa 0.5 como fallback e imprime un aviso.

    Args:
        model_path (str):               Ruta al .pth del modelo entrenado.
        tif_path   (str | list[str]):   Ruta/s al .tif de 2 canales.
        positions  (list | list[list]): Posiciones para un TIFF o lista de listas.
        patch_size (int):               Tamaño del recorte (default: 64).

    Returns:
        list of dict: 'tif_path', 'position', 'prediction', 'probabilities', 'class_name'.
    """
    import os
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MitochondriaContextCNN(num_channels=2, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ── Cargar umbral óptimo guardado en entrenamiento ────────────────────────
    umbral_path = model_path.replace('.pth', '_umbral_optimo.txt')
    if os.path.exists(umbral_path):
        with open(umbral_path) as f:
            umbral = float(f.read().strip())
        print(f"  Umbral óptimo cargado: {umbral:.4f}  (guardado en entrenamiento)")
    else:
        umbral = 0.5
        print(f"  [AVISO] No se encontró fichero de umbral óptimo. "
              f"Usando umbral por defecto = 0.5")
        print(f"  (Para obtener el umbral óptimo entrena el modelo al menos una vez "
              f"con el código actualizado)")

    # Guardar umbral para que classify_from_csv pueda pasarlo a evaluate_results
    classify_from_tif._ultimo_umbral = umbral

    if isinstance(tif_path, str):
        tif_path  = [tif_path]
        positions = [positions]

    if len(tif_path) != len(positions):
        raise ValueError(
            f"El número de TIFFs ({len(tif_path)}) y listas de posiciones "
            f"({len(positions)}) debe coincidir."
        )

    results = []
    for tif, pos_list in zip(tif_path, positions):
        print(f"Clasificando {len(pos_list)} partículas en: {tif}")
        canal_rojo, canal_verde = load_tif_image(tif)

        for pos in pos_list:
            pred_idx, probs, class_name = classify_single_particle(
                model, canal_rojo, canal_verde, pos, patch_size, device,
                umbral=umbral
            )
            results.append({
                'tif_path':      tif,
                'position':      pos,
                'prediction':    pred_idx,
                'probabilities': probs,
                'class_name':    class_name,
            })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Evaluación (modo con etiquetas)
# ──────────────────────────────────────────────────────────────────────────────

def calcular_metricas_umbral(scores, true_labels, umbral):
    '''
    Calcula TP, TN, FP, FN y métricas para un umbral concreto.

    Args:
        scores      (array-like): Probabilidades de clase positiva.
        true_labels (array-like): Etiquetas reales.
        umbral      (float):      Umbral de decisión a aplicar.

    Returns:
        dict con claves: TP, TN, FP, FN, recall, precision, f1, kappa.
    '''
    scores      = np.array(scores)
    true_labels = np.array(true_labels)

    preds = (scores >= umbral).astype(int)
    TP = int(((preds == 1) & (true_labels == 1)).sum())
    TN = int(((preds == 0) & (true_labels == 0)).sum())
    FP = int(((preds == 1) & (true_labels == 0)).sum())
    FN = int(((preds == 0) & (true_labels == 1)).sum())

    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    kappa_num = 2 * (TP * TN - FN * FP)
    kappa_den = (TP + FP) * (FP + TN) + (TP + FN) * (FN + TN)
    kappa     = kappa_num / kappa_den if kappa_den > 0 else 0.0

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'recall': recall, 'precision': precision,
            'f1': f1, 'kappa': kappa}


def calcular_metricas(scores, true_labels):
    '''
    Calcula TP, TN, FP, FN al umbral que maximiza F1 y devuelve todas
    las métricas: Recall, Precision, F1 y Cohen's Kappa.

    Args:
        scores      (array-like): Probabilidades de clase Borde.
        true_labels (array-like): Etiquetas reales (0=Borde, 1=Interior).

    Returns:
        dict con claves: TP, TN, FP, FN, recall, precision, f1, kappa,
                         umbral_optimo.
    '''
    from sklearn.metrics import precision_recall_curve
    scores      = np.array(scores)
    true_labels = np.array(true_labels)

    precisions, recalls, thresholds = precision_recall_curve(true_labels, scores)
    f1s = np.where((precisions[:-1] + recalls[:-1]) > 0,
                   2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
                   0)
    idx_opt       = int(np.argmax(f1s))
    umbral_optimo = float(thresholds[idx_opt])

    m = calcular_metricas_umbral(scores, true_labels, umbral_optimo)
    m['umbral_optimo'] = umbral_optimo
    return m



def plot_tabla_umbrales(scores, true_labels, save_dir=None, umbral_actual=None):
    """
    Genera una tabla consola con TP/TN/FP/FN/Recall/Precision/F1 para
    distintos umbrales, y guarda una curva Umbral vs métricas en disco.
    Útil para elegir el umbral óptimo sobre datos de inferencia etiquetados.

    Args:
        scores        (array-like): Probabilidades de clase Interior.
        true_labels   (array-like): Etiquetas reales (0=Borde, 1=Interior).
        save_dir      (str|None):   Directorio donde guardar la curva.
        umbral_actual (float|None): Umbral cargado del .txt, para marcarlo.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    scores      = np.array(scores)
    true_labels = np.array(true_labels)

    # ── Tabla por consola ─────────────────────────────────────────────────────
    umbrales_tabla = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                      0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  TABLA DE UMBRALES — SOBRE DATOS DE INFERENCIA")
    print(f"{sep}")
    print(f"  {'Umbral':>7}  {'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}  "
          f"{'Recall':>8}  {'Precision':>10}  {'F1':>8}")
    print(f"  {'─'*68}")
    for u in umbrales_tabla:
        preds = (scores >= u).astype(int)
        TP = int(((preds == 1) & (true_labels == 1)).sum())
        TN = int(((preds == 0) & (true_labels == 0)).sum())
        FP = int(((preds == 1) & (true_labels == 0)).sum())
        FN = int(((preds == 0) & (true_labels == 1)).sum())
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        marca = " ←" if umbral_actual is not None and abs(u - umbral_actual) < 0.026 else ""
        print(f"  {u:>7.2f}  {TP:>4}  {TN:>4}  {FP:>4}  {FN:>4}  "
              f"{rec:>8.4f}  {prec:>10.4f}  {f1:>8.4f}{marca}")
    print(f"{sep}\n")

    # ── Curva umbral sobre datos de inferencia ────────────────────────────────
    prec_th, rec_th, thresholds_th = precision_recall_curve(true_labels, scores)
    prec_th = prec_th[:-1]
    rec_th  = rec_th[:-1]
    f1_th   = np.where((prec_th + rec_th) > 0,
                       2 * prec_th * rec_th / (prec_th + rec_th), 0)

    if save_dir is None:
        save_dir = "resultados_clasificacion"
    os.makedirs(save_dir, exist_ok=True)
    path_umbral = os.path.join(save_dir, "curva_umbral_inferencia.png")

    fig_u, ax_u = plt.subplots(figsize=(8, 5))
    ax_u.plot(thresholds_th, rec_th,  color='steelblue',  lw=2, label='Recall (↑ = menos FN)')
    ax_u.plot(thresholds_th, prec_th, color='darkorange',  lw=2, label='Precision (↑ = menos FP)')
    ax_u.plot(thresholds_th, f1_th,   color='seagreen',    lw=2, label='F1')
    if umbral_actual is not None:
        ax_u.axvline(umbral_actual, color='gray', linestyle='--', alpha=0.8,
                     label=f'Umbral actual = {umbral_actual:.3f}')
    ax_u.set_xlim([0.0, 1.0]); ax_u.set_ylim([0.0, 1.05])
    ax_u.set_xlabel('Umbral de decisión (prob. Interior)', fontsize=12)
    ax_u.set_ylabel('Valor de la métrica', fontsize=12)
    ax_u.set_title('Umbral vs Recall / Precision / F1  —  Datos de inferencia\n'
                   'Mueve el umbral a la izquierda para más Recall (menos FN)', fontsize=11)
    ax_u.legend(loc='center right', fontsize=10)
    ax_u.grid(True, alpha=0.3)
    fig_u.tight_layout()
    fig_u.savefig(path_umbral, dpi=150, bbox_inches='tight')
    plt.close(fig_u)
    print(f"  Curva umbral (inferencia) guardada en: {path_umbral}")

def evaluate_results(results, true_labels, save_dir=None, umbral_actual=None):
    """
    Compara predicciones con etiquetas reales, imprime todas las métricas
    por consola y genera las curvas ROC y PR guardándolas en save_dir.

    Métricas calculadas:
      - AUC-ROC y Average Precision (independientes del umbral)
      - Recall, Precision, F1, Cohen's Kappa (al umbral que maximiza F1)

    Las curvas se guardan siempre en disco (en save_dir o en
    'resultados_clasificacion/' si save_dir es None). No se muestran
    métricas dentro de las gráficas — solo en consola.

    Args:
        results     (list of dict): Salida de classify_from_tif.
                    Cada dict debe tener 'prediction' y 'probabilities'.
        true_labels (list of int):  Etiquetas reales en el mismo orden.
        save_dir    (str | None):   Directorio donde guardar las figuras.

    Returns:
        dict: 'auc', 'pr_auc', 'recall', 'precision', 'f1', 'kappa'.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')

    from sklearn.metrics import (roc_curve, auc as sk_auc,
                                  precision_recall_curve)

    scores      = np.array([r['probabilities'][1] for r in results])
    true_labels = np.array(true_labels)

    # ── Curvas globales ───────────────────────────────────────────────────────
    fpr, tpr, _            = roc_curve(true_labels, scores)
    roc_auc                = sk_auc(fpr, tpr)
    precisions, recalls, _ = precision_recall_curve(true_labels, scores)
    pr_auc                 = sk_auc(recalls, precisions)   # igual que en entrenamiento

    # ── Métricas al umbral óptimo ─────────────────────────────────────────────
    m = calcular_metricas(scores, true_labels)

    # ── Métricas al umbral del modelo (el cargado del .txt) ──────────────────
    u_modelo = umbral_actual if umbral_actual is not None else 0.5
    m_modelo = calcular_metricas_umbral(scores, true_labels, u_modelo)

    # ── Consola ───────────────────────────────────────────────────────────────
    sep = "=" * 52
    # Bloque 1: umbral del modelo (lo que realmente clasificó la red)
    print(f"\n{sep}")
    print(f"  MÉTRICAS — UMBRAL DEL MODELO ({u_modelo:.4f})")
    print(f"  (son las métricas reales de lo que clasificó la red)")
    print(f"{sep}")
    print(f"  TP={m_modelo['TP']}  TN={m_modelo['TN']}  FP={m_modelo['FP']}  FN={m_modelo['FN']}")
    print(f"  {'─'*46}")
    print(f"  Recall         : {m_modelo['recall']:.4f}")
    print(f"  Precision      : {m_modelo['precision']:.4f}")
    print(f"  F1 score       : {m_modelo['f1']:.4f}")
    print(f"  Cohen's Kappa  : {m_modelo['kappa']:.4f}")
    print(f"  AUC-ROC        : {roc_auc:.4f}")
    print(f"  AUC-PR         : {pr_auc:.4f}  (área bajo curva PR)")
    print(f"{sep}")

    # Bloque 2: umbral óptimo F1 (referencia)
    print(f"\n{sep}")
    print(f"  MÉTRICAS — UMBRAL ÓPTIMO F1 ({m['umbral_optimo']:.4f})")
    print(f"  (referencia: qué métricas se obtendrían con este umbral)")
    print(f"{sep}")
    print(f"  TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")
    print(f"  {'─'*46}")
    print(f"  Recall         : {m['recall']:.4f}")
    print(f"  Precision      : {m['precision']:.4f}")
    print(f"  F1 score       : {m['f1']:.4f}")
    print(f"  Cohen's Kappa  : {m['kappa']:.4f}")
    print(f"{sep}\n")

    # ── Rutas de guardado ─────────────────────────────────────────────────────
    if save_dir is None:
        save_dir = "resultados_clasificacion"
    os.makedirs(save_dir, exist_ok=True)
    path_roc = os.path.join(save_dir, "curva_ROC.png")
    path_pr  = os.path.join(save_dir, "curva_PR.png")

    # ── Curva ROC ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    ax_roc.plot(fpr, tpr, color='steelblue', lw=2,
                label=f'Curva ROC  (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=1.2,
                linestyle='--', label='Clasificador aleatorio (AUC = 0.5)')
    ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR / Recall)', fontsize=12)
    ax_roc.set_title('Curva ROC — Evaluación del clasificador\n'
                     'Clasificador Borde / Interior', fontsize=13)
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
    baseline = true_labels.mean()
    ax_pr.axhline(y=baseline, color='gray', lw=1.2, linestyle='--',
                  label=f'Clasificador aleatorio (P = {baseline:.2f})')
    ax_pr.set_xlim([0.0, 1.0]); ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall (Sensibilidad)', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Curva PR — Evaluación del clasificador\n'
                    'Clasificador Borde / Interior', fontsize=13)
    ax_pr.legend(loc='upper right', fontsize=11)
    ax_pr.grid(True, alpha=0.3)
    fig_pr.tight_layout()
    fig_pr.savefig(path_pr, dpi=150, bbox_inches='tight')
    plt.close(fig_pr)
    print(f"  Curva PR  guardada en: {path_pr}")

    # ── Tabla de umbrales sobre datos de inferencia ──────────────────────────
    plot_tabla_umbrales(scores, true_labels,
                        save_dir=save_dir, umbral_actual=umbral_actual)

    return {'auc': roc_auc, 'pr_auc': pr_auc,
            'recall': m['recall'], 'precision': m['precision'],
            'f1': m['f1'], 'kappa': m['kappa']}


# ──────────────────────────────────────────────────────────────────────────────
# Guardado de resultados en CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_results_to_csv(csv_path, results_per_csv, aisladas_mask=None,
                         d2v=None, n_vecinas=None):
    """
    Añade columnas de resultados al CSV original y lo guarda con el sufijo
    '_clasificado'. Las partículas detectadas como aisladas reciben el valor
    'Aislada' en la columna 'clasificacion'; el resto recibe la predicción
    de la red junto con las probabilidades individuales de cada clase.

    Columnas añadidas:
      - clasificacion    : 'Borde', 'Interior' o 'Aislada'
      - prob_borde       : probabilidad de clase Borde [0,1]   (índice 0)
      - prob_interior    : probabilidad de clase Interior [0,1] (índice 1)
      - dist_verde_px    : distancia al verde (solo modo inferencia)
      - n_vecinas        : nº vecinas en radio (solo modo inferencia)

    También escribe un archivo de log '<nombre>_clasificado.log' en el mismo
    directorio con el detalle partícula a partícula.

    Args:
        csv_path        (str):             CSV original.
        results_per_csv (list[dict]):      Resultados de la red (solo no-aisladas).
        aisladas_mask   (np.ndarray bool): Máscara de aisladas sobre todas las filas.
        d2v             (np.ndarray):      Distancia al verde por partícula.
        n_vecinas       (np.ndarray):      Nº vecinas en radio por partícula.

    Returns:
        str: Ruta del archivo CSV guardado.
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    clasificacion  = []
    prob_borde     = []
    prob_interior  = []
    result_iter    = iter(results_per_csv)

    for i in range(len(df)):
        if aisladas_mask is not None and aisladas_mask[i]:
            clasificacion.append('Aislada')
            prob_borde.append(np.nan)     # aisladas no pasan por la red
            prob_interior.append(np.nan)
        else:
            r = next(result_iter)
            clasificacion.append(r['class_name'])
            prob_borde.append(round(float(r['probabilities'][0]), 4))
            prob_interior.append(round(float(r['probabilities'][1]), 4))

    df['clasificacion'] = clasificacion
    df['prob_borde']    = prob_borde
    df['prob_interior'] = prob_interior

    if d2v is not None:
        df['dist_verde_px'] = np.round(d2v, 2)
    if n_vecinas is not None:
        df['n_vecinas'] = n_vecinas

    p        = Path(csv_path)
    out_path = p.with_name(p.stem + '_clasificado' + p.suffix)
    df.to_csv(out_path, index=False)
    print(f"  Resultados guardados en: {out_path}")

    # ── Log partícula a partícula ─────────────────────────────────────────────
    log_path = out_path.with_suffix('.log')
    with open(log_path, 'w', encoding='utf-8') as lf:
        lf.write(f"Clasificación de: {csv_path}\n")
        lf.write(f"{'─'*70}\n")
        lf.write(f"{'#':>5}  {'y':>7}  {'x':>7}  {'clasificacion':>13}  "
                 f"{'P(Borde)':>11}  {'P(Interior)':>9}\n")
        lf.write(f"{'─'*70}\n")

        # Columnas y/x — buscar nombres case-insensitive
        cols = {c.lower(): c for c in df.columns}
        col_y = cols.get('y', 'y')
        col_x = cols.get('x', 'x')

        for i, row in df.iterrows():
            p_b  = f"{row['prob_borde']:.4f}"    if not pd.isna(row['prob_borde'])    else "  —   "
            p_i  = f"{row['prob_interior']:.4f}" if not pd.isna(row['prob_interior']) else "  —   "
            lf.write(f"{i+1:>5}  {row[col_y]:>7}  {row[col_x]:>7}  "
                     f"{row['clasificacion']:>13}  {p_b:>11}  {p_i:>9}\n")

        lf.write(f"{'─'*70}\n")
        n_borde   = (df['clasificacion'] == 'Borde').sum()
        n_noborde = (df['clasificacion'] == 'Interior').sum()
        n_aislada = (df['clasificacion'] == 'Aislada').sum()
        lf.write(f"Total: {len(df)}  |  Borde: {n_borde}  |  "
                 f"Interior: {n_noborde}  |  Aislada: {n_aislada}\n")

    print(f"  Log detallado guardado en: {log_path}")
    return str(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Función principal de clasificación
# ──────────────────────────────────────────────────────────────────────────────

def classify_from_csv(model_path, tif_path, csv_path, patch_size=64,
                      save_dir=None,
                      umbral_verde=None, umbral_dist=2.0,
                      usar_conectividad=False):
    """
    Clasifica partículas a partir de uno o varios pares TIFF + CSV.

    Comportamiento según el contenido del CSV:

    CON columna 'clase' (modo evaluación):
        - Las partículas etiquetadas como 'aislada' se excluyen directamente
          de la clasificación por la red (se marcan como 'Aislada' en el CSV
          de salida sin pasar por el modelo).
        - El resto se clasifica con la red y se evalúa contra las etiquetas
          reales: imprime por consola Recall, Precision, F1, Cohen's Kappa,
          AUC-ROC y Average Precision, y guarda las curvas ROC y PR en save_dir.

    SIN columna 'clase' (modo inferencia):
        - Se ejecuta el detector EDT de aisladas sobre cada imagen: se binariza
          el verde (Otsu si umbral_verde=None), se calcula la transformada de
          distancia al verde y una partícula se marca aislada si su distancia al
          verde supera el umbral θ (umbral_dist). Así se excluyen las aisladas.
        - El resto se clasifica con la red.
        - Al final se reporta cuántas partículas se excluyeron como aisladas
          en cada imagen y en total.

    En ambos casos el CSV de salida incluye la columna 'clasificacion' con el
    resultado ('Borde', 'Interior' o 'Aislada') para cada partícula, además de
    la columna auxiliar 'dist_verde_px' (distancia EDT al verde) en inferencia.

    Args:
        model_path   (str):             Ruta al .pth del modelo.
        tif_path     (str | list[str]): Ruta/s al .tif de 2 canales.
        csv_path     (str | list[str]): Ruta/s al CSV (con o sin columna 'clase').
        patch_size   (int):             Tamaño del recorte (default: 64).
        save_dir     (str | None):      Directorio donde guardar curva_ROC.png y
                                        curva_PR.png en modo evaluación.
                                        Si es None se usa 'resultados_clasificacion/'.
        umbral_verde (float | None):    Umbral de binarización del verde para la EDT.
                                        None -> Otsu automático por imagen (recomendado).
        umbral_dist  (float):           Umbral de distancia θ (px): aislada ⟺ d_g > θ.
                                        Valor fijo obtenido del barrido de detectar_aisladas.

    Returns:
        list of dict: Una entrada por partícula NO aislada, con claves:
                      'tif_path', 'position', 'prediction', 'probabilities',
                      'class_name' y, en modo evaluación, 'true_label'.
    """
    if isinstance(tif_path, str):
        tif_path = [tif_path]
    if isinstance(csv_path, str):
        csv_path = [csv_path]

    if len(tif_path) != len(csv_path):
        raise ValueError(
            f"El número de TIFFs ({len(tif_path)}) y CSVs ({len(csv_path)}) "
            "debe coincidir."
        )

    # ── Leer todos los CSVs: posiciones directamente del DataFrame ───────────
    # Se lee el DataFrame directamente (no load_labels_csv) para garantizar
    # que el orden (y, x) es idéntico al de evaluar_test y _generar_csvs_test.
    all_positions_orig = []  # posiciones (y, x) de TODAS las partículas (inc. aisladas)
    all_clases_raw     = []  # etiquetas texto originales (o None)
    has_labels         = []

    for csv in csv_path:
        df_tmp = pd.read_csv(csv, encoding='utf-8-sig')
        df_tmp.columns = [c.strip().lower() for c in df_tmp.columns]
        if 'y' not in df_tmp.columns or 'x' not in df_tmp.columns:
            raise ValueError(f"El CSV '{csv}' no tiene columnas 'y' y/o 'x'.")
        pos = list(zip(df_tmp['y'].astype(float), df_tmp['x'].astype(float)))
        clases_raw = (list(df_tmp['clase'].str.strip().str.lower())
                      if 'clase' in df_tmp.columns else None)
        all_positions_orig.append(pos)
        all_clases_raw.append(clases_raw)
        has_labels.append(clases_raw is not None)

    evaluation_mode = all(has_labels)
    if any(has_labels) and not evaluation_mode:
        print("AVISO: algunos CSVs tienen columna 'clase' y otros no. "
              "Se usa modo inferencia para todos.")
        evaluation_mode = False

    # ── Procesar cada par ─────────────────────────────────────────────────────
    all_results_red   = []   # solo resultados de partículas que pasan por la red
    all_true_labels   = []   # etiquetas reales de las que pasan por la red
    total_aisladas    = 0
    total_clasificadas = 0

    tif_paths_red  = []   # TIFFs a pasar a classify_from_tif
    pos_lists_red  = []   # posiciones a clasificar por la red

    # Guardar metadatos por CSV para el guardado posterior
    meta_por_csv = []   # lista de dicts con info de cada CSV

    for tif, csv, positions_orig, clases_raw in zip(
            tif_path, csv_path, all_positions_orig, all_clases_raw):

        canal_rojo, canal_verde = load_tif_image(tif)
        n_total = len(positions_orig)

        if evaluation_mode:
            # ── Con etiquetas: excluir aisladas etiquetadas ───────────────────
            mask_aisladas = np.array([c == 'aislada' for c in clases_raw])
            d2v       = None
            n_vecinas = None
        else:
            # ── Sin etiquetas: detectar aisladas con EDT al verde ─────────────
            print(f"\n  Detectando aisladas con EDT (verde) en: {tif}")
            mask_aisladas, d2v, n_vecinas = detectar_aisladas_EDT(
                canal_verde, positions_orig,
                canal_rojo=canal_rojo,
                umbral_verde=umbral_verde,        # None -> Otsu automático por imagen
                umbral_dist=umbral_dist,          # θ fijo (óptimo del barrido)
                usar_conectividad=usar_conectividad,
            )

        n_aisladas    = int(mask_aisladas.sum())
        pos_no_aisl   = [p for p, m in zip(positions_orig, mask_aisladas) if not m]
        total_aisladas    += n_aisladas
        total_clasificadas += len(pos_no_aisl)

        print(f"  {tif}: {n_total} partículas → "
              f"{n_aisladas} aisladas excluidas, "
              f"{len(pos_no_aisl)} pasan por la red.")

        tif_paths_red.append(tif)
        pos_lists_red.append(pos_no_aisl)

        # Etiquetas reales de las no-aisladas (solo en modo evaluación)
        if evaluation_mode:
            # Mismo mapeo que evaluar_test en entrenamiento.py
            CLASS_MAP = {'borde': 0, 'interior': 1, 'exterior': 1}
            true_lbl_no_aisl = [CLASS_MAP[c] for c, m in zip(clases_raw, mask_aisladas)
                                 if not m]
            all_true_labels.extend(true_lbl_no_aisl)

        meta_por_csv.append({
            'csv':           csv,
            'mask_aisladas': mask_aisladas,
            'd2v':           d2v,
            'n_vecinas':     n_vecinas,
            'n_no_aisl':     len(pos_no_aisl),
        })

    # ── Clasificar con la red todas las no-aisladas de todos los TIFFs ────────
    print(f"\n  Clasificando {total_clasificadas} partículas con la red neuronal...")
    all_results_red = classify_from_tif(model_path, tif_paths_red,
                                         pos_lists_red, patch_size)

    # ── Guardar CSV de resultados por archivo ─────────────────────────────────
    print()
    idx = 0
    for meta in meta_por_csv:
        n          = meta['n_no_aisl']
        res_csv    = all_results_red[idx: idx + n]
        save_results_to_csv(
            meta['csv'], res_csv,
            aisladas_mask=meta['mask_aisladas'],
            d2v=meta['d2v'],
            n_vecinas=meta['n_vecinas'],
        )
        idx += n

    # ── Resumen de aisladas excluidas ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  RESUMEN DE PARTÍCULAS AISLADAS EXCLUIDAS")
    print(f"{'─'*55}")
    for meta in meta_por_csv:
        n_aisl = int(meta['mask_aisladas'].sum())
        n_tot  = len(meta['mask_aisladas'])
        modo   = "etiqueta" if evaluation_mode else "detector EDT (verde)"
        print(f"  {Path(meta['csv']).name}: "
              f"{n_aisl}/{n_tot} excluidas ({modo})")
    print(f"  TOTAL: {total_aisladas} aisladas excluidas de "
          f"{total_aisladas + total_clasificadas} partículas.")
    print(f"{'─'*55}")

    # ── Modo evaluación: añadir true_label y calcular métricas ───────────────
    if evaluation_mode:
        for r, true_lbl in zip(all_results_red, all_true_labels):
            r['true_label'] = true_lbl

        print(f"\nModo evaluación: {len(all_results_red)} partículas "
              f"clasificadas por la red (aisladas excluidas previamente).")
        # Pasar el umbral cargado del .txt para marcarlo en la curva y la tabla
        umbral_cargado = getattr(classify_from_tif, "_ultimo_umbral", None)
        evaluate_results(all_results_red, all_true_labels,
                         save_dir=save_dir, umbral_actual=umbral_cargado)
    else:
        print(f"\nModo inferencia completado.")

    return all_results_red


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_PATH = "best_mito_classifier.pth"

    # ── Opción A: una sola imagen ─────────────────────────────────────────────
    TIF_PATH = "../datos_clasificar/SUb_02_10_orig.tif"
    CSV_PATH = "../datos_clasificar/SUb_02_10_orig.csv" 
    

    # ── Opción B: listas manuales ─────────────────────────────────────────────
    # TIF_PATH = ["datos/img1.tif", "datos/img2.tif"]
    # CSV_PATH = ["datos/img1.csv", "datos/img2.csv"]

    # ── Opción C: directorio completo ─────────────────────────────────────────
    #TIF_PATH, CSV_PATH = load_pairs_from_dir("data_augmentation")

    # Directorio donde guardar curva_ROC.png y curva_PR.png (modo evaluación)
    # None = se usa 'resultados_clasificacion/' por defecto
    SAVE_DIR = "resultados_clasificacion"

    classify_from_csv(
        MODEL_PATH, TIF_PATH, CSV_PATH,
        save_dir=SAVE_DIR,
        # Parámetros del detector EDT (solo se usan en modo inferencia, sin 'clase'):
        umbral_verde=None,      # None -> Otsu (umbral del verde automático por imagen)
        umbral_dist=16.0,        # θ fijo: pon aquí el óptimo de tu barrido
        usar_conectividad=False,
    )