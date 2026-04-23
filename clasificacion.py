import torch
import torch.nn.functional as F
import numpy as np

from funciones_auxiliares import load_tif_image, load_labels_csv, extract_patch_around_particle
from modelo_CNN import MitochondriaContextCNN

CLASS_NAMES = ['No borde', 'Borde'] #se cambia a clasificacion final binaria


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

    # Recorte de 2 canales → (2, patch_size, patch_size)
    patch = extract_patch_around_particle(canal_rojo, canal_verde, particle_center, patch_size)

    # Tensor [1, 2, H, W]
    patch_tensor = torch.FloatTensor(patch).unsqueeze(0).to(device)

    with torch.no_grad():
        output        = model(patch_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        prediction    = int(output.argmax(dim=1).cpu().numpy()[0])

    return prediction, probabilities, CLASS_NAMES[prediction]


def classify_from_tif(model_path, tif_path, positions, patch_size=64):
    """
    Carga el modelo entrenado y clasifica una lista de posiciones de partículas
    sobre un TIFF de 2 canales.

    Args:
        model_path (str):  Ruta al archivo .pth guardado durante el entrenamiento.
        tif_path   (str):  Ruta al archivo .tif de 2 canales.
        positions  (list): Lista de posiciones [(y1,x1), (y2,x2), ...].
        patch_size (int):  Tamaño del recorte (por defecto 64).

    Returns:
        list of dict: Una entrada por partícula con claves:
                      'position', 'prediction', 'probabilities', 'class_name'.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar modelo
    model = MitochondriaContextCNN(num_channels=2, num_classes=2) #nuevo numero de clases de salida
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Cargar imagen
    canal_rojo, canal_verde = load_tif_image(tif_path)

    results = []
    for pos in positions:
        pred_idx, probs, class_name = classify_single_particle(
            model, canal_rojo, canal_verde, pos, patch_size, device
        )
        results.append({
            'position':      pos,
            'prediction':    pred_idx,
            'probabilities': probs,
            'class_name':    class_name,
        })

    return results


def classify_from_csv(model_path, tif_path, csv_path, patch_size=64):
    """
    Versión de classify_from_tif que lee las posiciones directamente de un CSV.
    Modo puramente de inferencia: clasifica las partículas y devuelve resultados.
    No realiza ninguna evaluación (para eso está entrenamiento.py).

    Args:
        model_path (str): Ruta al archivo .pth.
        tif_path   (str): Ruta al archivo .tif de 2 canales.
        csv_path   (str): Ruta al CSV con columnas x, y (sin etiquetas).
        patch_size (int): Tamaño del recorte (por defecto 64).

    Returns:
        list of dict: Una entrada por partícula con claves:
                      'position', 'prediction', 'probabilities', 'class_name'.
    """
    positions, _ = load_labels_csv(csv_path)
    results      = classify_from_tif(model_path, tif_path, positions, patch_size)
    print(f"Clasificadas {len(results)} partículas.")
    return results


if __name__ == "__main__":
    MODEL_PATH = "best_mito_classifier.pth" #cambiar por archivo del modelo
    TIF_PATH   = "imagen.tif"      # Cambiar por nombre imagen
    CSV_PATH   = "etiquetas.csv"   # Cambiar por nombre archivo de datos

    results = classify_from_csv(MODEL_PATH, TIF_PATH, CSV_PATH)

    for r in results:
        print(f"  ({r['position'][0]:4d}, {r['position'][1]:4d}) → "
              f"{r['class_name']:8s}  "
              f"[NoBorde={r['probabilities'][0]:.2f} "
              f"Borde={r['probabilities'][1]:.2f}]")
