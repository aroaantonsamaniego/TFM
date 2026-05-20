import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

from funciones_auxiliares import (load_tif_image, load_labels_csv,
                                   extract_patch_around_particle,
                                   detectar_aisladas_geometrico,
                                   load_pairs_from_dir)
from modelo_CNN import MitochondriaContextCNN

CLASS_NAMES = ['No borde', 'Borde']


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
        umbral          (float): Umbral de decisión sobre prob. de clase Borde.

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

def calcular_metricas(scores, true_labels):
    '''
    Calcula TP, TN, FP, FN al umbral que maximiza F1 y devuelve todas
    las métricas: Recall, Precision, F1 y Cohen's Kappa.

    Args:
        scores      (array-like): Probabilidades de clase Borde.
        true_labels (array-like): Etiquetas reales (0=No borde, 1=Borde).

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

    preds = (scores >= umbral_optimo).astype(int)
    TP = int(((preds == 1) & (true_labels == 1)).sum())
    TN = int(((preds == 0) & (true_labels == 0)).sum())
    FP = int(((preds == 1) & (true_labels == 0)).sum())
    FN = int(((preds == 0) & (true_labels == 1)).sum())

    recall    = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    kappa_num = 2 * (TP * TN - FN * FP)
    kappa_den = (TP + FP) * (FP + TN) + (TP + FN) * (FN + TN)
    kappa     = kappa_num / kappa_den if kappa_den > 0 else 0.0

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'recall': recall, 'precision': precision,
            'f1': f1, 'kappa': kappa,
            'umbral_optimo': umbral_optimo}


def evaluate_results(results, true_labels, save_dir=None):
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
        dict: 'auc', 'avg_precision', 'recall', 'precision', 'f1', 'kappa'.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')

    from sklearn.metrics import (roc_curve, auc as sk_auc,
                                  precision_recall_curve,
                                  average_precision_score)

    scores      = np.array([r['probabilities'][1] for r in results])
    true_labels = np.array(true_labels)

    # ── Curvas globales ───────────────────────────────────────────────────────
    fpr, tpr, _            = roc_curve(true_labels, scores)
    roc_auc                = sk_auc(fpr, tpr)
    precisions, recalls, _ = precision_recall_curve(true_labels, scores)
    avg_precision          = average_precision_score(true_labels, scores)

    # ── Métricas al umbral óptimo ─────────────────────────────────────────────
    m = calcular_metricas(scores, true_labels)

    # ── Consola ───────────────────────────────────────────────────────────────
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  RESULTADOS DE EVALUACIÓN")
    print(f"{sep}")
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
    baseline = true_labels.mean()
    ax_pr.axhline(y=baseline, color='gray', lw=1.2, linestyle='--',
                  label=f'Clasificador aleatorio (P = {baseline:.2f})')
    ax_pr.set_xlim([0.0, 1.0]); ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall (Sensibilidad)', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Curva PR — Evaluación del clasificador\n'
                    'Clasificador Borde / No borde', fontsize=13)
    ax_pr.legend(loc='upper right', fontsize=11)
    ax_pr.grid(True, alpha=0.3)
    fig_pr.tight_layout()
    fig_pr.savefig(path_pr, dpi=150, bbox_inches='tight')
    plt.close(fig_pr)
    print(f"  Curva PR  guardada en: {path_pr}")

    return {'auc': roc_auc, 'avg_precision': avg_precision,
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
      - clasificacion    : 'Borde', 'No borde' o 'Aislada'
      - prob_no_borde    : probabilidad de clase No borde [0,1]
      - prob_borde       : probabilidad de clase Borde [0,1]
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
    prob_no_borde  = []
    prob_borde     = []
    result_iter    = iter(results_per_csv)

    for i in range(len(df)):
        if aisladas_mask is not None and aisladas_mask[i]:
            clasificacion.append('Aislada')
            prob_no_borde.append(np.nan)   # aisladas no pasan por la red
            prob_borde.append(np.nan)
        else:
            r = next(result_iter)
            clasificacion.append(r['class_name'])
            prob_no_borde.append(round(float(r['probabilities'][0]), 4))
            prob_borde.append(round(float(r['probabilities'][1]), 4))

    df['clasificacion'] = clasificacion
    df['prob_no_borde'] = prob_no_borde
    df['prob_borde']    = prob_borde

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
                 f"{'P(No borde)':>11}  {'P(Borde)':>9}\n")
        lf.write(f"{'─'*70}\n")

        # Columnas y/x — buscar nombres case-insensitive
        cols = {c.lower(): c for c in df.columns}
        col_y = cols.get('y', 'y')
        col_x = cols.get('x', 'x')

        for i, row in df.iterrows():
            p_nb = f"{row['prob_no_borde']:.4f}" if not pd.isna(row['prob_no_borde']) else "  —   "
            p_b  = f"{row['prob_borde']:.4f}"    if not pd.isna(row['prob_borde'])    else "  —   "
            lf.write(f"{i+1:>5}  {row[col_y]:>7}  {row[col_x]:>7}  "
                     f"{row['clasificacion']:>13}  {p_nb:>11}  {p_b:>9}\n")

        lf.write(f"{'─'*70}\n")
        n_borde   = (df['clasificacion'] == 'Borde').sum()
        n_noborde = (df['clasificacion'] == 'No borde').sum()
        n_aislada = (df['clasificacion'] == 'Aislada').sum()
        lf.write(f"Total: {len(df)}  |  Borde: {n_borde}  |  "
                 f"No borde: {n_noborde}  |  Aislada: {n_aislada}\n")

    print(f"  Log detallado guardado en: {log_path}")
    return str(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Función principal de clasificación
# ──────────────────────────────────────────────────────────────────────────────

def classify_from_csv(model_path, tif_path, csv_path, patch_size=64,
                      save_dir=None,
                      radio_grafo=100, max_vecinas=3, umbral_verde=0.01):
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
        - Se ejecuta el detector geométrico automático de aisladas sobre cada
          imagen (criterio C: sola Y lejos del verde) para identificar y
          excluir las partículas aisladas.
        - El resto se clasifica con la red.
        - Al final se reporta cuántas partículas se excluyeron como aisladas
          en cada imagen y en total.

    En ambos casos el CSV de salida incluye la columna 'clasificacion' con el
    resultado ('No borde', 'Borde' o 'Aislada') para cada partícula, además
    de columnas auxiliares 'dist_verde_px' y 'n_vecinas'.

    Args:
        model_path   (str):             Ruta al .pth del modelo.
        tif_path     (str | list[str]): Ruta/s al .tif de 2 canales.
        csv_path     (str | list[str]): Ruta/s al CSV (con o sin columna 'clase').
        patch_size   (int):             Tamaño del recorte (default: 64).
        save_dir     (str | None):      Directorio donde guardar curva_ROC.png y
                                        curva_PR.png en modo evaluación.
                                        Si es None se usa 'resultados_clasificacion/'.
        radio_grafo  (float):           Radio (px) para contar vecinas en modo inferencia.
        max_vecinas  (int):             Máx. vecinas para considerar aislada.
        umbral_verde (float):           Umbral de intensidad para definir verde.

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

    # ── Determinar modo según si todos los CSVs tienen etiquetas ─────────────
    all_positions_orig = []  # posiciones de TODAS las partículas (inc. aisladas)
    all_clases_raw     = []  # etiquetas texto originales (o None)
    has_labels         = []

    for csv in csv_path:
        pos, labels = load_labels_csv(csv)
        # Leer también clases_raw para saber cuáles son 'aislada'
        df_tmp = pd.read_csv(csv, encoding='utf-8-sig')
        df_tmp.columns = [c.strip().lower() for c in df_tmp.columns]
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
            # ── Sin etiquetas: detectar aisladas geométricamente ──────────────
            print(f"\n  Detectando aisladas geométricamente en: {tif}")
            mask_aisladas, d2v, n_vecinas = detectar_aisladas_geometrico(
                canal_verde, positions_orig,
                radio_grafo=radio_grafo,
                max_vecinas=max_vecinas,
                umbral_verde=umbral_verde,
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
            from funciones_auxiliares import CLASS_MAP
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
        modo   = "etiqueta" if evaluation_mode else "detector geométrico"
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
        evaluate_results(all_results_red, all_true_labels, save_dir=save_dir)
    else:
        print(f"\nModo inferencia completado.")

    return all_results_red


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_PATH = "best_mito_classifier.pth"

    # ── Opción A: una sola imagen ─────────────────────────────────────────────
    # TIF_PATH = "datos/SUb_01_2_merged.tif"
    # CSV_PATH = "datos/SUb_01_2_datos_training.csv"  # con 'clase' → evaluación
    # CSV_PATH = "datos/SUb_01_2_nuevos.csv"          # sin 'clase' → inferencia

    # ── Opción B: listas manuales ─────────────────────────────────────────────
    # TIF_PATH = ["datos/img1.tif", "datos/img2.tif"]
    # CSV_PATH = ["datos/img1.csv", "datos/img2.csv"]

    # ── Opción C: directorio completo ─────────────────────────────────────────
    TIF_PATH, CSV_PATH = load_pairs_from_dir("data_augmentation")

    # Directorio donde guardar curva_ROC.png y curva_PR.png (modo evaluación)
    # None = se usa 'resultados_clasificacion/' por defecto
    SAVE_DIR = "resultados_clasificacion"

    classify_from_csv(
        MODEL_PATH, TIF_PATH, CSV_PATH,
        save_dir=SAVE_DIR,
        # Parámetros del detector geométrico (solo se usan sin columna 'clase'):
        radio_grafo=100, max_vecinas=3, umbral_verde=0.01,
    )