import torch
import torch.nn.functional as F
import numpy as np

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
        positions = [positions]   # lista de posiciones → lista de listas

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
                'tif_path':    tif,
                'position':    pos,
                'prediction':  pred_idx,
                'probabilities': probs,
                'class_name':  class_name,
            })

    return results


def classify_from_csv(model_path, tif_path, csv_path, patch_size=64):
    """
    Versión de classify_from_tif que lee las posiciones directamente de CSV/s.
    Modo puramente de inferencia: clasifica las partículas y devuelve resultados.

    Acepta tanto rutas individuales (str) como listas de rutas (list).
    Si se pasan listas, deben tener la misma longitud y se emparejan por índice:
        tif_path[0] <-> csv_path[0]
        tif_path[1] <-> csv_path[1]
        ...

    Args:
        model_path (str):             Ruta al archivo .pth.
        tif_path   (str | list[str]): Ruta/s al archivo .tif de 2 canales.
        csv_path   (str | list[str]): Ruta/s al CSV con columnas x, y (sin etiquetas necesarias).
        patch_size (int):             Tamaño del recorte (por defecto 64).

    Returns:
        list of dict: Una entrada por partícula con claves:
                      'tif_path', 'position', 'prediction', 'probabilities', 'class_name'.
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

    # Leer posiciones de cada CSV
    all_positions = []
    for csv in csv_path:
        pos, _ = load_labels_csv(csv)
        all_positions.append(pos)

    # classify_from_tif ya acepta listas
    results = classify_from_tif(model_path, tif_path, all_positions, patch_size)
    print(f"Clasificadas {len(results)} partículas en total ({len(tif_path)} imagen/es).")
    return results


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_PATH = "best_mito_classifier.pth"   # <-- ruta al modelo entrenado

    # ── Opción A: una sola imagen ─────────────────────────────────────────────
    # TIF_PATH = "imagen.tif"
    # CSV_PATH = "etiquetas.csv"

    # ── Opción B: varias imágenes (listas del mismo tamaño, emparejadas) ──────
    TIF_PATH = ["imagen1.tif","imagen2.tif","imagen3.tif"]
    CSV_PATH = ["etiquetas1.csv","etiquetas2.csv","etiquetas3.csv"]

    results = classify_from_csv(MODEL_PATH, TIF_PATH, CSV_PATH)

    for r in results:
        print(f"  {r['tif_path']} | "
              f"({r['position'][0]:4d}, {r['position'][1]:4d}) → "
              f"{r['class_name']:8s}  "
              f"[NoBorde={r['probabilities'][0]:.2f} "
              f"Borde={r['probabilities'][1]:.2f}]")
