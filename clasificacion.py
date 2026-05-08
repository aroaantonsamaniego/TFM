import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

from funciones_auxiliares import load_tif_image, load_labels_csv, extract_patch_around_particle
from modelo_CNN import MitochondriaContextCNN

CLASS_NAMES = ['No borde', 'Borde']


def classify_single_particle(model, canal_rojo, canal_verde, particle_center,
                              patch_size=64, device=None):
    """
    Clasifica UNA partícula recortando ambos canales de la imagen real
    centrados en sus coordenadas y pasando el tensor [1, 2, 64, 64] a la red.

    Args:
        model          : Red neuronal entrenada (MitochondriaContextCNN).
        canal_rojo     (np.ndarray): Canal rojo normalizado (H, W).
        canal_verde    (np.ndarray): Canal verde normalizado (H, W).
        particle_center(tuple): Coordenadas (y, x) de la partícula.
        patch_size     (int): Tamaño del recorte (por defecto 64).
        device         : Dispositivo torch. Si None, se detecta automáticamente.

    Returns:
        tuple: (prediction_idx, probabilities, class_name)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patch = extract_patch_around_particle(canal_rojo, canal_verde, particle_center, patch_size)
    patch_tensor = torch.FloatTensor(patch).unsqueeze(0).to(device)

    with torch.no_grad():
        output        = model(patch_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        prediction    = int(output.argmax(dim=1).cpu().numpy()[0])

    return prediction, probabilities, CLASS_NAMES[prediction]


def classify_from_tif(model_path, tif_path, positions, patch_size=64):
    """
    Carga el modelo entrenado y clasifica una lista de posiciones de partículas
    sobre uno o varios TIFFs de 2 canales.

    Acepta tanto una ruta individual (str) como una lista de rutas (list).
    Si se pasa una lista de TIFFs, también debe pasarse una lista de listas de
    posiciones del mismo tamaño, donde positions[i] corresponde a tif_path[i].

    Args:
        model_path (str):              Ruta al archivo .pth guardado durante el entrenamiento.
        tif_path   (str | list[str]):  Ruta/s al archivo .tif de 2 canales.
        positions  (list | list[list]): Lista de posiciones [(y1,x1), ...] para un solo TIFF,
                                        o lista de listas [[(y1,x1),...], [(y1,x1),...]] para varios.
        patch_size (int):              Tamaño del recorte (por defecto 64).

    Returns:
        list of dict: Una entrada por partícula con claves:
                      'tif_path', 'position', 'prediction', 'probabilities', 'class_name'.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar modelo una sola vez
    model = MitochondriaContextCNN(num_channels=2, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Normalizar a listas para un procesamiento uniforme
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
                model, canal_rojo, canal_verde, pos, patch_size, device
            )
            results.append({
                'tif_path':      tif,
                'position':      pos,
                'prediction':    pred_idx,
                'probabilities': probs,
                'class_name':    class_name,
            })

    return results


def plot_roc_curve(scores, labels, save_path=None):
    """
    Calcula y representa la curva ROC sobre los resultados de clasificación.

    Args:
        scores    (list of float): Probabilidades de clase Borde para cada muestra.
        labels    (list of int):   Etiquetas reales (0=No borde, 1=Borde).
        save_path (str | None):    Ruta opcional para guardar la figura en disco.

    Returns:
        float: Valor del AUC.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'Curva ROC  (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.2,
            linestyle='--', label='Clasificador aleatorio (AUC = 0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    ax.set_title('Curva ROC - Evaluación del modelo\nClasificador Borde / No borde', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Curva ROC guardada en: {save_path}")

    plt.show()
    return roc_auc


def evaluate_results(results, true_labels, roc_save_path=None):
    """
    Compara las predicciones del modelo con las etiquetas reales y muestra
    métricas de evaluación: accuracy, reporte por clase, matriz de confusión
    y curva ROC.

    Args:
        results       (list of dict): Salida de classify_from_csv con etiquetas.
                                      Cada dict debe tener 'prediction' y 'probabilities'.
        true_labels   (list of int):  Etiquetas reales en el mismo orden que results.
        roc_save_path (str | None):   Ruta opcional para guardar la curva ROC en disco.

    Returns:
        dict: Métricas calculadas con claves 'accuracy', 'auc', 'confusion_matrix'.
    """
    predictions = [r['prediction']        for r in results]
    scores      = [r['probabilities'][1]  for r in results]  # prob. clase Borde

    # Accuracy global
    correct  = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / len(true_labels)

    # Matriz de confusión y reporte por clase
    cm     = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions,
                                   target_names=CLASS_NAMES, zero_division=0)

    # Curva ROC y AUC
    roc_auc = plot_roc_curve(scores, true_labels, save_path=roc_save_path)

    # Imprimir resumen
    print("\n" + "="*55)
    print("          RESULTADOS DE EVALUACIÓN")
    print("="*55)
    print(f"  Total partículas evaluadas : {len(true_labels)}")
    print(f"  Accuracy                   : {accuracy:.4f}  ({correct}/{len(true_labels)})")
    print(f"  AUC-ROC                    : {roc_auc:.4f}")
    print("-"*55)
    print("  Matriz de confusión:")
    print(f"               Pred NoBorde  Pred Borde")
    print(f"  Real NoBorde   {cm[0,0]:>6d}        {cm[0,1]:>6d}")
    print(f"  Real Borde     {cm[1,0]:>6d}        {cm[1,1]:>6d}")
    print("-"*55)
    print("  Reporte por clase:")
    print(report)
    print("="*55)

    return {
        'accuracy':         accuracy,
        'auc':              roc_auc,
        'confusion_matrix': cm,
    }


def save_results_to_csv(csv_path, results_per_csv):
    """
    Añade la columna 'clasificacion' al CSV original y lo guarda con el sufijo
    '_clasificado' en el mismo directorio. Si el CSV ya tenía la columna
    'clasificacion', la sobreescribe.

    Args:
        csv_path        (str):        Ruta al CSV original.
        results_per_csv (list[dict]): Resultados de classify_from_tif correspondientes
                                      a ese CSV, en el mismo orden que las filas del CSV.

    Returns:
        str: Ruta del archivo guardado.
    """
    df = pd.read_csv(csv_path)
    df['clasificacion'] = [r['class_name'] for r in results_per_csv]

    p        = Path(csv_path)
    out_path = p.with_name(p.stem + '_clasificado' + p.suffix)
    df.to_csv(out_path, index=False)
    print(f"  Resultados guardados en: {out_path}")
    return str(out_path)


def classify_from_csv(model_path, tif_path, csv_path, patch_size=64, roc_save_path=None):
    """
    Clasifica partículas a partir de uno o varios pares TIFF + CSV y guarda
    los resultados añadiendo la columna 'clasificacion' a cada CSV de entrada,
    con el sufijo '_clasificado' en el nombre del archivo de salida.

    Si los CSV contienen la columna 'clase', activa automáticamente el modo
    evaluación: compara las predicciones con las etiquetas reales y muestra
    accuracy, matriz de confusión y curva ROC.
    Si los CSV no tienen 'clase', solo hace inferencia.

    Acepta tanto rutas individuales (str) como listas de rutas (list).
    Si se pasan listas, deben tener la misma longitud y se emparejan por índice:
        tif_path[0] <-> csv_path[0]
        tif_path[1] <-> csv_path[1]
        ...

    Args:
        model_path    (str):             Ruta al archivo .pth.
        tif_path      (str | list[str]): Ruta/s al archivo .tif de 2 canales.
        csv_path      (str | list[str]): Ruta/s al CSV con columnas x, y (y opcionalmente clase).
        patch_size    (int):             Tamaño del recorte (por defecto 64).
        roc_save_path (str | None):      Ruta para guardar la curva ROC (solo en modo evaluación).

    Returns:
        list of dict: Una entrada por partícula con claves:
                      'tif_path', 'position', 'prediction', 'probabilities', 'class_name'.
                      En modo evaluación, cada dict incluye además 'true_label'.
    """
    # Normalizar a listas
    if isinstance(tif_path, str):
        tif_path = [tif_path]
    if isinstance(csv_path, str):
        csv_path = [csv_path]

    if len(tif_path) != len(csv_path):
        raise ValueError(
            f"El número de TIFFs ({len(tif_path)}) y CSVs ({len(csv_path)}) debe coincidir."
        )

    # Leer posiciones y etiquetas de cada CSV
    all_positions  = []
    all_labels_raw = []
    has_labels     = []

    for csv in csv_path:
        pos, labels = load_labels_csv(csv)
        all_positions.append(pos)
        all_labels_raw.append(labels)
        has_labels.append(labels is not None)

    # Modo evaluación solo si TODOS los CSVs tienen etiquetas
    evaluation_mode = all(has_labels)

    if any(has_labels) and not evaluation_mode:
        print("AVISO: algunos CSVs tienen la columna 'clase' y otros no. "
              "Se omiten las etiquetas y se hace solo inferencia.")

    # Clasificación
    results = classify_from_tif(model_path, tif_path, all_positions, patch_size)

    # ── Guardar CSV clasificados ───────────────────────────────────────────────
    # Repartir los resultados globales por CSV según cuántas partículas tiene cada uno
    print()
    idx = 0
    for csv, pos_list in zip(csv_path, all_positions):
        n                = len(pos_list)
        results_for_csv  = results[idx: idx + n]
        save_results_to_csv(csv, results_for_csv)
        idx += n

    # ── Modo evaluación ────────────────────────────────────────────────────────
    if evaluation_mode:
        true_labels = []
        for labels in all_labels_raw:
            true_labels.extend(labels)

        for r, true_lbl in zip(results, true_labels):
            r['true_label'] = true_lbl

        print(f"\nModo evaluación activado: {len(results)} partículas clasificadas "
              f"y comparadas con sus etiquetas reales.")
        evaluate_results(results, true_labels, roc_save_path=roc_save_path)

    else:
        print(f"\nClasificadas {len(results)} partículas en total ({len(tif_path)} imagen/es).")

    return results


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_PATH = "best_mito_classifier.pth"   

    TIF_PATH = ["datos/SUb_01_2_merged.tif"]
    CSV_PATH = ["datos/SUb_01_2_datos_training.csv"]

    ROC_SAVE_PATH = "roc_curve_eval.png"   # None para no guardar en disco

    classify_from_csv(MODEL_PATH, TIF_PATH, CSV_PATH, roc_save_path=ROC_SAVE_PATH)