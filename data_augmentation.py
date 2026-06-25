import os
from pathlib import Path

import albumentations as A
import pandas as pd
import tifffile as tiff

# =========================================================
# CONFIGURACIÓN
# =========================================================

# Carpeta con imágenes Y CSVs originales (mismo directorio)
INPUT_DIR = "definitivos"

# Carpeta de salida
OUTPUT_DIR = "data_augmentation_clase"

# Extensión de imágenes
IMAGE_EXT = ".tif"

# Crear carpeta de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# TRANSFORMACIONES
# =========================================================

TRANSFORMS = {
    "orig": A.Compose(
        [],
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        )
    ),

    "rot90": A.Compose(
        [A.RandomRotate90(p=1)],
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        )
    ),

    "flipH": A.Compose(
        [A.HorizontalFlip(p=1)],
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        )
    ),

    "flipV": A.Compose(
        [A.VerticalFlip(p=1)],
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        )
    ),

    "flipHV": A.Compose(
        [
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1)
        ],
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        )
    ),

    "rot_aleatoria_30": A.Compose(
        [
            # limit=30 significa que girará un ángulo aleatorio entre -30 y +30 grados
            # border_mode=0 rellena los huecos vacíos de las esquinas con color negro
            A.Rotate(limit=30, border_mode=0, value=0, p=1)
        ],
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        )
    ),
}

# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def process_sample(image_path, csv_path):

    # -----------------------------------------------------
    # 1. Leer imagen y sus metadatos de color (ImageJ)
    # -----------------------------------------------------
    with tiff.TiffFile(str(image_path)) as tif:
        image = tif.asarray()
        
        # Intentamos extraer los metadatos que le dan el color en Fiji
        try:
            metadata = tif.imagej_metadata
        except:
            metadata = None

    # Albumentations necesita el formato (Alto, Ancho, Canales)
    # Si la imagen viene de Fiji, suele ser (Canales, Alto, Ancho). Ejemplo: (2, 958, 1611)
    is_channel_first = False
    if image.ndim == 3 and image.shape[0] < image.shape[-1]: 
        is_channel_first = True
        image = image.transpose(1, 2, 0) # La convertimos a (Alto, Ancho, Canales)

    # -----------------------------------------------------
    # 2. Leer CSV
    # -----------------------------------------------------
    df = pd.read_csv(csv_path)
    points = list(zip(df["X"], df["Y"]))
    clase = df["Clase"] if "Clase" in df.columns else None
    base_name = image_path.stem

    # -----------------------------------------------------
    # 3. Aplicar transformaciones
    # -----------------------------------------------------
    for aug_name, transform in TRANSFORMS.items():
        augmented = transform(
            image=image,
            keypoints=points
        )

        new_image = augmented["image"]
        new_points = augmented["keypoints"]

        # -------------------------------------------------
        # 4. Actualizar CSV
        # -------------------------------------------------
        new_df = pd.DataFrame({
            "X": [p[0] for p in new_points],
            "Y": [p[1] for p in new_points]
        })

        # Copiar la columna Clase del original si existe
        if clase is not None:
            new_df["Clase"] = clase.values

        # -------------------------------------------------
        # 5. Guardar imagen restaurando el formato y color
        # -------------------------------------------------
        # Devolvemos la imagen al formato original de Fiji (Canales, Alto, Ancho)
        if is_channel_first:
            new_image_to_save = new_image.transpose(2, 0, 1)
        else:
            new_image_to_save = new_image

        out_img_name = f"{base_name}_{aug_name}.tif"
        out_img_path = Path(OUTPUT_DIR) / out_img_name

        # Guardamos pasándole los metadatos originales para que recupere el RGB
        tiff.imwrite(
            str(out_img_path), 
            new_image_to_save, 
            imagej=True, 
            metadata=metadata
        )

        # -------------------------------------------------
        # 6. Guardar CSV
        # -------------------------------------------------
        out_csv_name = f"{base_name}_{aug_name}.csv"
        out_csv_path = Path(OUTPUT_DIR) / out_csv_name
        new_df.to_csv(out_csv_path, index=False)

        print(f"[OK] {out_img_name}")

# =========================================================
# RECORRER DATASET
# =========================================================

image_files = sorted(Path(INPUT_DIR).glob(f"*{IMAGE_EXT}"))
print(f"Directorio de búsqueda: {Path(INPUT_DIR).absolute()}")
print(f"Imágenes encontradas: {len(image_files)}")

for image_path in image_files:

    base_name = image_path.stem

    # El CSV se busca en el mismo directorio que la imagen
    csv_path = image_path.with_suffix(".csv")

    if not csv_path.exists():
        print(f"[WARNING] CSV no encontrado para {base_name}")
        continue

    process_sample(image_path, csv_path)

print("\nDataset augmentado completado.")