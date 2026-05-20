import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
import pandas as pd
from scipy.spatial import cKDTree


# ──────────────────────────────────────────────────────────────────────────────
# Carga de imagen
# ──────────────────────────────────────────────────────────────────────────────

def load_tif_image(tif_path):
    '''
    Carga un archivo .tif de 2 canales (rojo: estaticas, verde: elipticas).

    Args:
        tif_path (str): Ruta al archivo .tif.

    Returns:
        tuple: (canal_rojo, canal_verde) — cada uno es un np.ndarray (H, W)
               con valores normalizados en [0.0, 1.0].
    '''
    image = tifffile.imread(tif_path)

    if image.ndim == 3 and image.shape[2] == 2:
        image = np.moveaxis(image, -1, 0)  # (H, W, 2) → (2, H, W)

    if image.shape[0] != 2:
        raise ValueError(f"Se esperaban 2 canales, pero la imagen tiene forma {image.shape}")

    canal_rojo  = image[0].astype(np.float32)
    canal_verde = image[1].astype(np.float32)

    def normalize(channel):
        vmin, vmax = channel.min(), channel.max()
        if vmax > vmin:
            return (channel - vmin) / (vmax - vmin)
        return channel

    return normalize(canal_rojo), normalize(canal_verde)


# ──────────────────────────────────────────────────────────────────────────────
# Carga de CSV
# ──────────────────────────────────────────────────────────────────────────────

CLASS_MAP = {'interior': 0, 'exterior': 0, 'aislada': 0, 'borde': 1}

def load_labels_csv(csv_path):
    '''
    Carga el CSV con coordenadas de partículas. Soporta dos formatos:
    - Entrenamiento (3 columnas):  x, y, clase
    - Clasificación (2 columnas):  x, y

    Args:
        csv_path (str): Ruta al archivo .csv.

    Returns:
        tuple: (positions, labels)
               - positions: lista de tuplas [(y1, x1), (y2, x2), ...]
               - labels: lista de enteros o None si no hay columna 'clase'.
    '''
    print(f"--- Procesando archivo: {csv_path} ---")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = [c.strip().lower() for c in df.columns]

    if not {'x', 'y'}.issubset(df.columns):
        raise ValueError("El CSV debe tener al menos las columnas: x, y")

    positions = list(zip(df['y'].astype(float).astype(int),
                         df['x'].astype(float).astype(int)))

    if 'clase' in df.columns:
        df['clase'] = df['clase'].str.strip().str.lower()
        unknown = set(df['clase']) - set(CLASS_MAP.keys())
        if unknown:
            raise ValueError(f"Etiquetas desconocidas en el CSV: {unknown}. "
                             f"Usa: {list(CLASS_MAP.keys())}")
        labels = [CLASS_MAP[c] for c in df['clase']]
    else:
        labels = None

    return positions, labels


# ──────────────────────────────────────────────────────────────────────────────
# Filtrado de partículas aisladas
# ──────────────────────────────────────────────────────────────────────────────

def filtrar_aisladas_etiquetadas(positions, labels, clases_raw):
    '''
    Elimina del dataset las partículas etiquetadas como 'aislada'.
    Se usa cuando el CSV tiene columna 'clase'.

    Args:
        positions  (list of (y,x)): Coordenadas de todas las partículas.
        labels     (list of int):   Etiquetas numéricas (0=no borde, 1=borde).
        clases_raw (list of str):   Etiquetas de texto originales del CSV.

    Returns:
        tuple: (positions_filtradas, labels_filtradas, n_eliminadas)
    '''
    mask = [c != 'aislada' for c in clases_raw]
    pos_filtradas = [p for p, m in zip(positions, mask) if m]
    lbl_filtradas = [l for l, m in zip(labels, mask) if m]
    n_eliminadas  = sum(1 for m in mask if not m)
    return pos_filtradas, lbl_filtradas, n_eliminadas


def detectar_aisladas_geometrico(canal_verde, positions,
                                  radio_grafo=100, max_vecinas=3,
                                  umbral_verde=0.01):
    '''
    Detecta partículas aisladas automáticamente usando dos criterios geométricos
    combinados (criterio C: sola Y lejos del verde):

      1. Número de vecinas en radio_grafo <= max_vecinas
         (la partícula tiene pocas o ninguna partícula cerca).
      2. Distancia al píxel verde más cercano >= min_dist_verde
         (la partícula no está sobre ni cerca de la estructura mitocondrial).

    El umbral min_dist_verde se calcula automáticamente como el percentil 75
    de las distancias al verde de todas las partículas, de forma que se adapta
    a cada imagen sin necesidad de ajuste manual.

    Args:
        canal_verde  (np.ndarray): Canal verde normalizado (H, W).
        positions    (list of (y,x)): Coordenadas de las partículas.
        radio_grafo  (float): Radio en px para contar vecinas (default: 100).
        max_vecinas  (int):   Máximo de vecinas permitidas para ser aislada (default: 3).
        umbral_verde (float): Umbral de intensidad para definir estructura verde (default: 0.01).

    Returns:
        tuple: (mask_aisladas, d2v, n_vecinas)
               - mask_aisladas (np.ndarray bool): True = clasificada como aislada.
               - d2v           (np.ndarray float): Distancia al verde de cada partícula.
               - n_vecinas     (np.ndarray int):   Nº vecinas en radio_grafo.
    '''
    coords = np.array([(x, y) for y, x in positions], dtype=np.float32)
    n = len(coords)

    # ── Distancia al verde más cercano ────────────────────────────────────────
    ys_v, xs_v = np.where(canal_verde > umbral_verde)
    if len(ys_v) == 0:
        d2v = np.full(n, np.inf)
    else:
        tree_verde = cKDTree(np.column_stack([xs_v, ys_v]))
        d2v, _     = tree_verde.query(coords)

    # ── Número de vecinas en radio_grafo ──────────────────────────────────────
    tree_part = cKDTree(coords)
    n_vecinas = np.array([len(tree_part.query_ball_point(c, radio_grafo)) - 1
                          for c in coords])

    # ── Umbral de distancia al verde adaptativo (percentil 75) ───────────────
    min_dist_verde = float(np.percentile(d2v, 75))
    mask_aisladas  = (n_vecinas <= max_vecinas) & (d2v >= min_dist_verde)

    return mask_aisladas, d2v, n_vecinas


# ──────────────────────────────────────────────────────────────────────────────
# Extracción de parches
# ──────────────────────────────────────────────────────────────────────────────

def extract_patch_around_particle(canal_rojo, canal_verde, center_yx, patch_size=32):
    '''
    Extrae un recorte cuadrado de ambos canales centrado en una partícula.
    Usa padding 'reflect' siempre que el recorte tenga tamaño suficiente,
    y cae back a 'constant' (ceros) cuando la partícula está en el borde
    extremo de la imagen y el recorte quedaría vacío en alguna dimensión.

    Args:
        canal_rojo  (np.ndarray): Canal rojo (H, W), valores en [0,1].
        canal_verde (np.ndarray): Canal verde (H, W), valores en [0,1].
        center_yx   (tuple): Coordenadas (y, x) del centro.
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

        pad_y1 = patch_size // 2 - cy           if cy < patch_size // 2      else 0
        pad_y2 = patch_size // 2 - (h - cy - 1) if cy > h - patch_size // 2  else 0
        pad_x1 = patch_size // 2 - cx           if cx < patch_size // 2      else 0
        pad_x2 = patch_size // 2 - (w - cx - 1) if cx > w - patch_size // 2  else 0

        crop = channel[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros((patch_size, patch_size), dtype=np.float32)

        can_reflect = (crop.shape[0] >= max(pad_y1, pad_y2, 1) and
                       crop.shape[1] >= max(pad_x1, pad_x2, 1))
        mode = 'reflect' if can_reflect else 'constant'

        crop = np.pad(crop, [(pad_y1, pad_y2), (pad_x1, pad_x2)], mode=mode)
        return crop[:patch_size, :patch_size].astype(np.float32)

    cy, cx  = int(center_yx[0]), int(center_yx[1])
    patch_r = _crop(canal_rojo,  cy, cx)
    patch_g = _crop(canal_verde, cy, cx)
    return np.stack([patch_r, patch_g], axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ParticleDataset(Dataset):
    '''
    Dataset para entrenamiento supervisado.

    Args:
        patches (np.ndarray): Array (N, 2, H, W).
        labels  (list/array): Etiquetas enteras (0=no borde, 1=borde).
    '''
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels  = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = torch.FloatTensor(self.patches[idx])
        label = torch.LongTensor([self.labels[idx]])
        return patch, label


# ──────────────────────────────────────────────────────────────────────────────
# Construcción del dataset desde CSV
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset_from_csv(tif_path, csv_path, patch_size=32):
    '''
    Carga TIFFs y CSVs, filtra las partículas aisladas (etiquetadas como
    'aislada') y construye un ParticleDataset con el resto.

    Acepta rutas individuales (str) o listas de rutas (list).

    Args:
        tif_path   (str | list[str]): Ruta/s al .tif de 2 canales.
        csv_path   (str | list[str]): Ruta/s al CSV con columnas y, x, clase.
        patch_size (int):             Tamaño del recorte (default: 64).

    Returns:
        ParticleDataset
    '''
    if isinstance(tif_path, str):
        tif_path = [tif_path]
    if isinstance(csv_path, str):
        csv_path = [csv_path]

    if len(tif_path) != len(csv_path):
        raise ValueError(
            f"El número de TIFFs ({len(tif_path)}) y CSVs ({len(csv_path)}) "
            "debe coincidir."
        )

    all_patches      = []
    all_labels       = []
    total_aisladas   = 0
    total_particulas = 0

    for i, (tif, csv) in enumerate(zip(tif_path, csv_path)):
        print(f"  Cargando par {i+1}/{len(tif_path)}: {tif} + {csv}")
        canal_rojo, canal_verde = load_tif_image(tif)

        df = pd.read_csv(csv, encoding='utf-8-sig')
        df.columns = [c.strip().lower() for c in df.columns]

        if 'clase' not in df.columns:
            raise ValueError(
                f"El CSV '{csv}' no contiene la columna 'clase'. "
                "El CSV de entrenamiento debe incluir: x, y, clase"
            )

        df['clase'] = df['clase'].str.strip().str.lower()
        unknown = set(df['clase']) - set(CLASS_MAP.keys())
        if unknown:
            raise ValueError(f"Etiquetas desconocidas en '{csv}': {unknown}")

        total_particulas += len(df)

        # ── Filtrar aisladas etiquetadas ──────────────────────────────────────
        df_filtrado    = df[df['clase'] != 'aislada'].reset_index(drop=True)
        n_aisladas     = len(df) - len(df_filtrado)
        total_aisladas += n_aisladas

        if n_aisladas > 0:
            print(f"    Eliminadas {n_aisladas} partículas aisladas (etiquetadas). "
                  f"Quedan {len(df_filtrado)}.")

        positions = list(zip(df_filtrado['y'].astype(float).astype(int),
                             df_filtrado['x'].astype(float).astype(int)))
        labels    = [CLASS_MAP[c] for c in df_filtrado['clase']]

        for pos in positions:
            patch = extract_patch_around_particle(canal_rojo, canal_verde, pos, patch_size)
            all_patches.append(patch)

        all_labels.extend(labels)

    all_patches = np.stack(all_patches, axis=0)
    counts      = np.bincount(np.array(all_labels), minlength=2)

    print(f"\nDataset construido: {len(all_labels)} partículas "
          f"({total_aisladas} aisladas eliminadas de {total_particulas} totales) | "
          f"Distribución: no_borde={counts[0]}, borde={counts[1]}")

    return ParticleDataset(all_patches, all_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Carga automática de pares desde directorio
# ──────────────────────────────────────────────────────────────────────────────

def load_pairs_from_dir(directory, tif_ext=".tif", csv_ext=".csv"):
    '''
    Busca en un directorio todos los .tif y empareja cada uno con su CSV
    del mismo nombre base. Los pares incompletos se omiten con aviso.

    Args:
        directory (str): Ruta a la carpeta.
        tif_ext   (str): Extensión de imágenes (default: ".tif").
        csv_ext   (str): Extensión de CSVs (default: ".csv").

    Returns:
        tuple: (tif_paths, csv_paths) — dos listas del mismo tamaño.
    '''
    from pathlib import Path

    directory = Path(directory)
    tif_files = sorted(directory.glob(f"*{tif_ext}"))

    tif_paths, csv_paths = [], []

    for tif in tif_files:
        csv = tif.with_suffix(csv_ext)
        if csv.exists():
            tif_paths.append(str(tif))
            csv_paths.append(str(csv))
        else:
            print(f"[WARNING] Sin CSV para {tif.name}, se omite.")

    print(f"Directorio: {directory.absolute()}")
    print(f"Pares encontrados: {len(tif_paths)}")
    return tif_paths, csv_paths


# ──────────────────────────────────────────────────────────────────────────────
# Visualización de batches
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES_VIS = {0: 'No borde', 1: 'Borde'}


def visualize_batch(patches, labels, batch_idx=0, save_path=None):
    '''
    Visualiza todas las imágenes de un batch en una cuadrícula.

    Cada imagen se muestra con dos subpaneles: canal rojo (estáticas) arriba
    y canal verde (elípticas) abajo. El título de cada columna indica la
    etiqueta de clase de esa partícula.

    Args:
        patches   (torch.Tensor | np.ndarray): Batch de parches (N, 2, H, W).
        labels    (torch.Tensor | np.ndarray): Etiquetas del batch (N,) o (N, 1).
        batch_idx (int): Índice del batch (para el título de la figura).
        save_path (str | None): Ruta donde guardar la figura. None = no guardar.

    Returns:
        matplotlib.figure.Figure: La figura generada.
    '''
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if hasattr(patches, 'numpy'):
        patches = patches.numpy()
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()

    labels = np.array(labels).flatten()
    n      = len(labels)

    fig = plt.figure(figsize=(max(2 * n, 6), 5), constrained_layout=True)
    fig.suptitle(f'Batch {batch_idx}  —  {n} partículas',
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.05, wspace=0.15)

    canal_labels = ['Rojo\n(estáticas)', 'Verde\n(elípticas)']
    cmaps        = ['Reds', 'Greens']

    for col in range(n):
        patch     = patches[col]
        etiqueta  = CLASS_NAMES_VIS.get(int(labels[col]), str(int(labels[col])))
        color_tit = '#c0392b' if labels[col] == 1 else '#2980b9'

        for row in range(2):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(patch[row], cmap=cmaps[row], vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])

            if col == 0:
                ax.set_ylabel(canal_labels[row], fontsize=8,
                              rotation=0, labelpad=45, va='center')
            if row == 0:
                ax.set_title(etiqueta, fontsize=9, color=color_tit,
                             fontweight='bold', pad=3)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Batch {batch_idx} guardado en: {save_path}")

    plt.show()
    return fig


def visualize_loader(loader, n_batches=1, save_dir=None):
    '''
    Visualiza los primeros n_batches batches de un DataLoader.

    Args:
        loader    (DataLoader): DataLoader de entrenamiento o validación.
        n_batches (int): Número de batches a mostrar (default: 1).
        save_dir  (str | None): Directorio donde guardar las figuras.
                  None = no guardar. Ej: "visualizacion_batches/".
    '''
    import matplotlib.pyplot as plt
    from pathlib import Path

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for batch_idx, (patches, labels) in enumerate(loader):
        if batch_idx >= n_batches:
            break
        save_path = (str(Path(save_dir) / f"batch_{batch_idx:03d}.png")
                     if save_dir else None)
        visualize_batch(patches, labels, batch_idx=batch_idx, save_path=save_path)
        plt.close('all')