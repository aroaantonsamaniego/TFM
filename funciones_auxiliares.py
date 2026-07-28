import numpy as np
import tifffile
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

# torch solo se necesita para el Dataset de entrenamiento (ParticleDataset).
# Se importa de forma opcional para que este modulo comun pueda usarse tambien
# desde el script clasico (clasificacion_EDT.py) sin PyTorch instalado.
try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_DISPONIBLE = True
except ImportError:
    Dataset = object
    _TORCH_DISPONIBLE = False

# Elemento estructurante de conectividad para el cierre morfologico del rojo
_ST = np.ones((3, 3), dtype=bool)


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

CLASS_MAP = {'interior': 1, 'aislada': 1, 'borde': 0}

def load_labels_csv(csv_path):
    '''
    Carga el CSV con coordenadas de trayectorias. Soporta dos formatos:
    - Entrenamiento (3 columnas):  x, y, clase
    - Clasificacion (2 columnas):  x, y

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
# Filtrado de trayectorias aisladas
# ──────────────────────────────────────────────────────────────────────────────

def filtrar_aisladas_etiquetadas(positions, labels, clases_raw):
    '''
    Elimina del dataset las trayectorias etiquetadas como 'aislada'.
    Se usa cuando el CSV tiene columna 'clase'.

    Args:
        positions  (list of (y,x)): Coordenadas de todas las trayectorias.
        labels     (list of int):   Etiquetas numericas (0=borde, 1=interior).
        clases_raw (list of str):   Etiquetas de texto originales del CSV.

    Returns:
        tuple: (positions_filtradas, labels_filtradas, n_eliminadas)
    '''
    mask = [c != 'aislada' for c in clases_raw]
    pos_filtradas = [p for p, m in zip(positions, mask) if m]
    lbl_filtradas = [l for l, m in zip(labels, mask) if m]
    n_eliminadas  = sum(1 for m in mask if not m)
    return pos_filtradas, lbl_filtradas, n_eliminadas


def construir_territorios(canal_rojo, filas, cols, factor_rojo=0.25):
    '''
    Construye un TERRITORIO por trayectoria mediante watershed del canal rojo
    sembrado con los centroides anotados (un territorio por trayectoria, con la
    etiqueta i+1). Portado verbatim de la Etapa 1 de clasificacion_EDT.py para
    que el modulo de la CNN sea independiente de ese script.

    Args:
        canal_rojo  (np.ndarray): Canal rojo (H, W). Puede venir normalizado: el
                                  umbral de Otsu escala igual que la imagen, asi
                                  que la mascara resultante es identica a la del
                                  canal crudo (transformacion monotona).
        filas       (np.ndarray): Coordenada y (int) de cada centroide.
        cols        (np.ndarray): Coordenada x (int) de cada centroide.
        factor_rojo (float):      t_r = factor_rojo * Otsu(rojo>0) para el rojo.

    Returns:
        tuple: (lab, t_r) — lab es el mapa de territorios (int32, 0=fondo,
               i+1=territorio de la trayectoria i); t_r el umbral usado.
    '''
    nz  = canal_rojo[canal_rojo > 0]
    t_r = float(threshold_otsu(nz)) * factor_rojo if nz.size else 0.0
    Mr  = canal_rojo > t_r
    Mr[filas, cols] = True
    Mr  = ndi.binary_closing(Mr, structure=_ST)
    sem = np.zeros(canal_rojo.shape, dtype=np.int32)
    sem[filas, cols] = np.arange(1, len(filas) + 1)
    lab = watershed(-ndi.distance_transform_edt(Mr), sem, mask=Mr)
    return lab, t_r


def detectar_aisladas_EDT(canal_verde, positions, canal_rojo,
                          umbral_dist=0.0,
                          umbral_verde=0.0,
                          factor_rojo=0.25):
    '''
    Detecta trayectorias aisladas
    Criterio (territorios watershed + d_min sobre el soporte del verde):

      1. M_g   = canal_verde > umbral_verde          (soporte del verde; el
                                                      optimo empirico es t_g = 0,
                                                      es decir M_g = {I_g > 0}).
      2. D_out = distance_transform_edt(~M_g)         (distancia de cada pixel a
                                                      la traza verde mas cercana).
      3. Por cada trayectoria i se construye su territorio rojo R_i (watershed
         del rojo sembrado con los centroides) y se toma
             d_min(i) = min_{p in R_i} D_out(p).
      4. aislada  <=>  d_min(i) > umbral_dist (theta). Optimo empirico theta = 0.

    Medir d_min sobre todo el territorio (no solo en el centroide) y usar la
    mascara de soporte {I_g > 0} es lo que hace a este criterio la
    version optima.

    NOTA sobre la normalizacion: el criterio es invariante a la normalizacion
    min-max de load_tif_image, porque el fondo verde es exactamente 0 (la
    mascara {I_g > 0} no cambia), el umbral de Otsu del rojo escala igual que la
    imagen (misma mascara) y las distancias son puramente geometricas (en px).

    Args:
        canal_verde  (np.ndarray):       Canal verde (H, W).
        positions    (list of (y,x)):    Coordenadas de las trayectorias.
        canal_rojo   (np.ndarray):       Canal rojo (H, W). Se usa para construir
                                         los territorios watershed.
        umbral_dist  (float):            Umbral theta (px). aislada <=> d_min > theta.
        umbral_verde (float | None):     t_g del soporte del verde M_g={I_g>t_g}.
                                         None se interpreta como 0.0 (=> {I_g>0}).
        factor_rojo  (float):            t_r = factor_rojo * Otsu(rojo>0).

    Returns:
        tuple: (mask_aisladas, d_min, None)
               - mask_aisladas (np.ndarray bool): True = aislada.
               - d_min         (np.ndarray float): distancia minima al verde por
                                                   trayectoria (columna
                                                   'dist_verde_px' en el CSV).
               - None: la Etapa 1 no calcula vecinas (la columna 'n_vecinas' se
                       omite automaticamente en el CSV de salida).
    '''
    if canal_rojo is None:
        raise ValueError(
            "detectar_aisladas_EDT requiere 'canal_rojo' para construir los "
            "territorios watershed de la Etapa 1."
        )

    if umbral_verde is None:
        umbral_verde = 0.0        # soporte del verde M_g = {I_g > 0}

    n    = len(positions)
    H, W = canal_verde.shape
    filas = np.clip(np.array([int(p[0]) for p in positions]), 0, H - 1)
    cols  = np.clip(np.array([int(p[1]) for p in positions]), 0, W - 1)

    # ── Mascara de soporte del verde y campo de distancia al verde ────────────
    Mg1   = canal_verde > umbral_verde
    D_out = (ndi.distance_transform_edt(~Mg1) if Mg1.any()
             else np.full(canal_verde.shape, np.inf))

    # ── Territorios watershed (un territorio por trayectoria) ─────────────────
    lab, _ = construir_territorios(canal_rojo, filas, cols, factor_rojo)
    idx    = np.arange(1, n + 1)
    area   = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(int)

    # ── d_min: distancia minima al verde sobre todo el territorio ─────────────
    if np.isinf(D_out).all():
        d_min = np.full(n, np.inf)
    else:
        d_min = np.asarray(ndi.minimum(D_out, lab, index=idx), dtype=float)

    # Territorios vacios (sin pixeles asignados): se usa la distancia del propio
    # centroide como respaldo, igual que en clasificar_edt.
    vac = area == 0
    if vac.any():
        d_min[vac] = D_out[filas[vac], cols[vac]]

    # ── Criterio de la Etapa 1 ────────────────────────────────────────────────
    mask_aisladas = d_min > umbral_dist

    return mask_aisladas, d_min, None

# ──────────────────────────────────────────────────────────────────────────────
# Extraccion de parches
# ──────────────────────────────────────────────────────────────────────────────

def extract_patch_around_particle(canal_rojo, canal_verde, center_yx, patch_size=32):
    '''
    Extrae un recorte cuadrado de ambos canales centrado en una trayectoria.
    Usa padding 'reflect' siempre que el recorte tenga tamanyo suficiente,
    y cae back a 'constant' (ceros) cuando la trayectoria esta en el borde
    extremo de la imagen y el recorte quedaria vacio en alguna dimension.

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
        labels  (list/array): Etiquetas enteras (0=borde, 1=interior).
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
# Construccion del dataset desde CSV
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset_from_csv(tif_path, csv_path, patch_size=32):
    '''
    Carga TIFFs y CSVs, filtra las trayectorias aisladas (etiquetadas como
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
            print(f"    Eliminadas {n_aisladas} trayectorias aisladas (etiquetadas). "
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

    print(f"\nDataset construido: {len(all_labels)} trayectorias "
          f"({total_aisladas} aisladas eliminadas de {total_particulas} totales) | "
          f"Distribución: borde={counts[0]}, interior={counts[1]}")

    return ParticleDataset(all_patches, all_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Carga automatica de pares desde directorio
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
# Visualizacion de patches individuales
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES_VIS = {0: 'Borde', 1: 'Interior'}


def visualize_patch(patch, label, patch_idx=0, save_dir=None, save_tiff=True):
    '''
    Visualiza un unico parche con sus dos canales solapados (overlay RGB)
    y opcionalmente lo guarda como figura PNG y/o TIFF de 2 canales para Fiji.

    La visualizacion overlay mapea:
      - Canal rojo  (estáticas)  → rojo
      - Canal verde (elípticas)  → verde
    ambos superpuestos en una imagen RGB para apreciar la colocalizacion.

    Los TIFF se guardan con forma (2, H, W) y dtype float32, listos para
    abrirse en Fiji con "Image > Color > Make Composite".

    Args:
        patch     (np.ndarray): Parche de forma (2, H, W), valores en [0, 1].
        label     (int):        Etiqueta de clase (0=borde, 1=interior).
        patch_idx (int):        Indice del parche (para nombres de archivo y titulo).
        save_dir  (str | None): Directorio donde guardar los archivos.
                                None = no guardar nada.
        save_tiff (bool):       Si True, guarda tambien un TIFF de 2 canales
                                compatible con Fiji (default: True).

    Returns:
        matplotlib.figure.Figure: La figura generada.
    '''
    import matplotlib.pyplot as plt
    from pathlib import Path

    if hasattr(patch, 'numpy'):
        patch = patch.numpy()
    patch = np.array(patch, dtype=np.float32)

    etiqueta  = CLASS_NAMES_VIS.get(int(label), str(int(label)))
    color_tit = '#c0392b' if int(label) == 0 else '#2980b9'

    # ── Construir imagen RGB overlay ─────────────────────────────────────────
    h, w  = patch.shape[1], patch.shape[2]
    rgb   = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = patch[0]   # canal rojo  → R
    rgb[..., 1] = patch[1]   # canal verde → G
    # canal B = 0 (no hay tercer canal)

    # ── Figura con tres paneles: rojo | verde | overlay ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), constrained_layout=True)
    fig.suptitle(f'Parche {patch_idx}  —  {etiqueta}',
                 fontsize=11, fontweight='bold', color=color_tit)

    axes[0].imshow(patch[0], cmap='Reds',   vmin=0, vmax=1)
    axes[0].set_title('Rojo\n(estáticas)',   fontsize=8)

    axes[1].imshow(patch[1], cmap='Greens', vmin=0, vmax=1)
    axes[1].set_title('Verde\n(elípticas)', fontsize=8)

    axes[2].imshow(rgb, vmin=0, vmax=1)
    axes[2].set_title('Overlay\n(R+G)',     fontsize=8)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    # ── Guardado ──────────────────────────────────────────────────────────────
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)

        # PNG de la figura
        png_path = out / f"patch_{patch_idx:04d}_{etiqueta.lower()}.png"
        fig.savefig(str(png_path), dpi=120, bbox_inches='tight')
        print(f"  Parche {patch_idx} PNG guardado en: {png_path}")

        # TIFF de 2 canales para Fiji
        if save_tiff:
            tif_path = out / f"patch_{patch_idx:04d}_{etiqueta.lower()}.tif"
            # shape (2, H, W), float32 normalizado [0,1]
            tifffile.imwrite(str(tif_path), patch, imagej=True,
                             metadata={'axes': 'CYX'})
            print(f"  Parche {patch_idx} TIFF guardado en: {tif_path}")

    plt.show()
    return fig


def visualize_loader(loader, n_patches=10, save_dir=None, save_tiff=True):
    '''
    Visualiza los primeros n_patches parches de un DataLoader,
    generando una figura individual por parche.

    Itera sobre los batches y extrae parches uno a uno hasta alcanzar
    el numero solicitado.

    Args:
        loader    (DataLoader): DataLoader de entrenamiento o validacion.
        n_patches (int):        Número total de parches a visualizar (default: 10).
        save_dir  (str | None): Directorio donde guardar figuras PNG y TIFFs.
                                None = no guardar. Ej: "visualizacion_patches/".
        save_tiff (bool):       Si True, guarda tambien TIFFs de 2 canales
                                compatibles con Fiji (default: True).
    '''
    import matplotlib.pyplot as plt

    count = 0
    for patches, labels in loader:
        if count >= n_patches:
            break

        if hasattr(patches, 'numpy'):
            patches = patches.numpy()
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()

        labels = np.array(labels).flatten()

        for i in range(len(labels)):
            if count >= n_patches:
                break
            visualize_patch(patches[i], labels[i],
                            patch_idx=count,
                            save_dir=save_dir,
                            save_tiff=save_tiff)
            plt.close('all')
            count += 1

    print(f"\nVisualizacion completada: {count} parches procesados.")