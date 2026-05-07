import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
import pandas as pd


def load_tif_image(tif_path):
    '''
    Carga un archivo .tif de 2 canales (rojo: estaticas ,verde: elipticas)

    Args:
        tif_path (str): Ruta al archivo .tif.

    Returns:
        tuple: (canal_rojo, canal_verde) — cada uno es un np.ndarray (H, W)
               con valores normalizados en [0.0, 1.0]
    '''
    image = tifffile.imread(tif_path)  # Forma esperada: (2, H, W) o (H, W, 2)

    # Normalizar orden de ejes a (2, H, W) para pytorch
    if image.ndim == 3 and image.shape[2] == 2:
        image = np.moveaxis(image, -1, 0)  # (H, W, 2) → (2, H, W)

    if image.shape[0] != 2:
        raise ValueError(f"Se esperaban 2 canales, pero la imagen tiene forma {image.shape}")

    canal_rojo  = image[0].astype(np.float32)
    canal_verde = image[1].astype(np.float32)

    # Normalizar cada canal a [0, 1] independientemente
    def normalize(channel): #esto puede que funcione mal en algun momento si hay un pixel muuy fuera de rango
        vmin, vmax = channel.min(), channel.max()
        if vmax > vmin:
            return (channel - vmin) / (vmax - vmin)
        return channel
    
    return normalize(canal_rojo), normalize(canal_verde)


CLASS_MAP = {'interior': 0, 'exterior': 0,'aislada':0, 'borde': 1}  # binario: 0=No borde, 1=Borde

def load_labels_csv(csv_path):
    '''
    Carga el CSV con coordenadas de particulas. Soporta dos formatos:
    - Entrenamiento (3 columnas):
        x,y,clase
        340,120,interior   ← se convierte a 0 (No borde)
        150,200,borde      ← se convierte a 1 (Borde)
        220,310,exterior   ← se convierte a 0 (No borde)
        220,310,aislada   ← se convierte a 0 (No borde)

    - Clasificacion (2 columnas, sin etiquetas):
        x,y
        340,120
        150,200

    Args:
        csv_path (str): Ruta al archivo .csv.

    Returns:
        tuple: (positions, labels)
               - positions: lista de tuplas [(y1, x1), (y2, x2), ...]  ← orden interno siempre (y, x)
               - labels: lista de enteros [0, 1, 2, ...] o None si el CSV no tiene etiquetas.
    '''
    print(f"--- Procesando archivo: {csv_path} ---") # ver que archivo esta leyendo para detectar fallos
    df = pd.read_csv(csv_path)

    # Aceptar cabeceras en mayusculas o minusculas
    df.columns = [c.strip().lower() for c in df.columns]

    if not {'x', 'y'}.issubset(df.columns):
        raise ValueError("El CSV debe tener al menos las columnas: x, y")

    # CSV viene como x,y pero internamente siempre usamos (y, x)
    #positions = list(zip(df['y'].astype(int), df['x'].astype(int)))
    positions = list(zip(df['y'].astype(float).astype(int), df['x'].astype(float).astype(int)))

    # Etiquetas: solo si la columna 'clase' existe (modo entrenamiento/evaluacion)
    if 'clase' in df.columns:
        df['clase'] = df['clase'].str.strip().str.lower()
        unknown = set(df['clase']) - set(CLASS_MAP.keys())
        if unknown:
            raise ValueError(f"Etiquetas desconocidas en el CSV: {unknown}. "
                             f"Usa: {list(CLASS_MAP.keys())}")
        labels = [CLASS_MAP[c] for c in df['clase']]
    else:
        labels = None  # Modo clasificacion: no hay etiquetas

    return positions, labels


def extract_patch_around_particle(canal_rojo, canal_verde, center_yx, patch_size=64):
    '''
    Extrae un recorte cuadrado de AMBOS canales centrado en una particula
    y los devuelve apilados como un array (2, patch_size, patch_size).

    Args:
        canal_rojo  (np.ndarray): Canal rojo de la imagen (H, W), valores en [0,1].
        canal_verde (np.ndarray): Canal verde de la imagen (H, W), valores en [0,1].
        center_yx   (tuple): Coordenadas (y, x) del centro de la particula.
        patch_size  (int): Tamaño del lado del recorte.

    Returns:
        np.ndarray: Recorte de forma (2, patch_size, patch_size).
    '''
    def _crop(channel, cy, cx):
        h, w = channel.shape
        y1 = max(0, cy - patch_size // 2)
        y2 = min(h, cy + patch_size // 2)
        x1 = max(0, cx - patch_size // 2)
        x2 = min(w, cx + patch_size // 2)

        pad_y1 = patch_size // 2 - cy           if cy < patch_size // 2        else 0
        pad_y2 = patch_size // 2 - (h - cy - 1) if cy > h - patch_size // 2   else 0
        pad_x1 = patch_size // 2 - cx           if cx < patch_size // 2        else 0
        pad_x2 = patch_size // 2 - (w - cx - 1) if cx > w - patch_size // 2   else 0

        crop = channel[y1:y2, x1:x2]
        crop = np.pad(crop, [(pad_y1, pad_y2), (pad_x1, pad_x2)], mode='reflect')
        return crop[:patch_size, :patch_size].astype(np.float32)

    cy, cx = int(center_yx[0]), int(center_yx[1]) 
    patch_r = _crop(canal_rojo,  cy, cx)
    patch_g = _crop(canal_verde, cy, cx)

    return np.stack([patch_r, patch_g], axis=0)  # (2, H, W)


class ParticleDataset(Dataset):
    '''
    Dataset para entrenamiento supervisado.
    Almacena los recortes de 2 canales y sus etiquetas.

    Args:
        patches (np.ndarray): Array (N, 2, H, W) con los recortes.
        labels  (list/array): Etiquetas enteras (0=no borde, 1=borde).
    '''
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels  = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = torch.FloatTensor(self.patches[idx])   # [2, H, W]
        label = torch.LongTensor([self.labels[idx]])   # [1]
        return patch, label


def build_dataset_from_csv(tif_path, csv_path, patch_size=64):
    '''
    Carga el TIFF y el CSV, extrae los recortes de 2 canales para cada
    partícula y devuelve un ParticleDataset listo para el DataLoader.

    Acepta tanto rutas individuales (str) como listas de rutas (list).
    Si se pasan listas, deben tener la misma longitud y se emparejan por índice:
        tif_path[0] <-> csv_path[0]
        tif_path[1] <-> csv_path[1]
        ...

    Args:
        tif_path   (str | list[str]): Ruta/s al archivo .tif de 2 canales.
        csv_path   (str | list[str]): Ruta/s al CSV con columnas y, x, clase.
        patch_size (int):             Tamaño del recorte (por defecto 64).

    Returns:
        ParticleDataset
    '''
    # Normalizar a listas para un procesamiento uniforme
    if isinstance(tif_path, str):
        tif_path = [tif_path]
    if isinstance(csv_path, str):
        csv_path = [csv_path]

    if len(tif_path) != len(csv_path):
        raise ValueError(
            f"El número de TIFFs ({len(tif_path)}) y CSVs ({len(csv_path)}) debe coincidir. "
            "Cada TIFF debe tener su CSV de etiquetas correspondiente."
        )

    all_patches = []
    all_labels  = []

    for i, (tif, csv) in enumerate(zip(tif_path, csv_path)):
        print(f"  Cargando par {i+1}/{len(tif_path)}: {tif} + {csv}")
        canal_rojo, canal_verde = load_tif_image(tif)
        positions, labels       = load_labels_csv(csv)

        if labels is None:
            raise ValueError(
                f"El CSV '{csv}' no contiene la columna 'clase'. "
                "El CSV de entrenamiento debe incluir: x, y, clase"
            )

        for pos in positions:
            patch = extract_patch_around_particle(canal_rojo, canal_verde, pos, patch_size)
            all_patches.append(patch)

        all_labels.extend(labels)

    all_patches = np.stack(all_patches, axis=0)  # (N_total, 2, H, W)
    counts      = np.bincount(np.array(all_labels), minlength=2)

    print(f"Dataset construido: {len(all_labels)} partículas en total "
          f"({len(tif_path)} imagen/es) | "
          f"Distribución: no_borde={counts[0]}, borde={counts[1]}")

    return ParticleDataset(all_patches, all_labels)
