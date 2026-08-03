"""
Extractor de posiciones de trayectorias estaticas (X, Y) -> CSV.

Pasandole un directorio donde estan almacenadas las imagenes tif de dos canales,
genera el correspondiente archivo de posciones para cada una de ellas y los guarda
con exactante el mismo nombre de la imagen

Uso
---
1) Edita la CONFIGURACION de abajo (como minimo DIRECTORIO_IMAGENES) y ejecuta:
       python extraer_posiciones.py
2) O pasa el directorio por linea de comandos (sobrescribe la config):
       python extraer_posiciones.py ../datos/imagenes_nuevas
       python extraer_posiciones.py ../datos/imagenes_nuevas --salida ../datos/posiciones
       python extraer_posiciones.py ../datos/imagenes_nuevas --sobrescribir
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

# =========================================================
# CONFIGURACION
# =========================================================
# Directorio con las imagenes TIFF de dos canales a procesar.
DIRECTORIO_IMAGENES = "../datos/prueba_extraer"

# Carpeta donde escribir los CSV.
#   - None  -> se escribe cada CSV JUNTO a su TIFF (recomendado: asi
#              load_pairs_from_dir del clasificador los empareja sin mas).
#   - ruta  -> se escriben todos en esa carpeta con el nombre '<stem>.csv'.
DIRECTORIO_SALIDA = None

# Sufijo opcional para el nombre del CSV ('imagen.tif' -> 'imagen<SUFIJO>.csv').
# Dejar en "" para que el nombre base coincida con el TIFF (necesario para que
# load_pairs_from_dir del clasificador lo encuentre).
SUFIJO_CSV = ""

# Extensiones de imagen que se procesan.
EXTENSIONES = (".tif", ".tiff")

# Seguridad: si el CSV de salida YA existe, por defecto NO se sobrescribe.
# Esto evita destruir anotaciones previas si por error apuntas a una carpeta ya
# anotada. Pon True (o usa --sobrescribir) para regenerarlos.
SOBRESCRIBIR = False

# X, Y como enteros (coherente con el recorte de patches y con la EDT del
# clasificador). Igual que EXPORT_INT del anotador.
EXPORTAR_ENTEROS = True

# ── Parametros de deteccion ───────────────────
METODO_DETECCION  = "maximos"   # "maximos" (picos locales) | "centroides" (blobs)
DET_SIGMA         = 1.0         # suavizado gaussiano previo (px); 0 = sin suavizar
DET_UMBRAL        = None        # None: Otsu automatico sobre el canal rojo
DET_MIN_DISTANCIA = 3           # ("maximos")   separacion minima entre picos (px)
DET_AREA_MIN      = 2           # ("centroides") area minima de un blob (px)
DET_AREA_MAX      = None        # ("centroides") area maxima (None = sin limite)
# =========================================================


# =========================================================
# DETECCION 
# =========================================================
def canal_estaticas(image):
    """Devuelve el canal de estaticas (canal 0) como 2D float [0, 1].

    Replica la convencion de load_tif_image: el canal 0 es el rojo (estaticas).
    Acepta (H,W), (2,H,W)/(C,H,W) channel-first o (H,W,C) channel-last.
    """
    img = image
    if img.ndim == 2:
        ch = img
    else:
        # channel-first -> channel-last si el primer eje es el mas pequenyo
        if img.ndim == 3 and img.shape[0] < img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))
        ch = img[..., 0]   # canal 0 = estaticas, igual que load_tif_image

    ch = ch.astype(float)
    mn, mx = ch.min(), ch.max()
    return (ch - mn) / (mx - mn) if mx > mn else np.zeros_like(ch)


def detectar_estaticas(canal, metodo=METODO_DETECCION, umbral=DET_UMBRAL,
                       sigma=DET_SIGMA, min_distancia=DET_MIN_DISTANCIA,
                       area_min=DET_AREA_MIN, area_max=DET_AREA_MAX):
    """Localiza las trayectorias estaticas sobre el canal rojo normalizado.

    Args:
        canal        (np.ndarray): Canal de estaticas (H, W) en [0, 1].
        metodo       (str): "maximos"   -> picos locales (skimage.peak_local_max).
                            "centroides" -> umbral + componentes conexas (centroide).
        umbral       (float|None): Umbral de intensidad. None -> Otsu automatico.
        sigma        (float): Suavizado gaussiano previo (px). 0 = sin suavizar.
        min_distancia(int):   ("maximos") separacion minima entre picos (px).
        area_min     (int):   ("centroides") area minima de un blob (px).
        area_max     (int|None): ("centroides") area maxima (None = sin limite).

    Returns:
        list[(x, y)]: coordenadas enteras (columna, fila) de cada estatica.
    """
    try:
        from scipy import ndimage as ndi
    except Exception as e:
        raise ImportError("La deteccion automatica necesita scipy "
                          "(pip install scipy).") from e

    suav = ndi.gaussian_filter(canal, sigma) if (sigma and sigma > 0) else canal

    # ── Umbral (fijo o Otsu) ─────────────────────────────────────────────────
    if umbral is None:
        try:
            from skimage.filters import threshold_otsu
            t = float(threshold_otsu(suav))
        except Exception:
            t = float(suav.mean())
    else:
        t = float(umbral)

    mask = suav > t

    puntos = []
    if metodo == "maximos":
        try:
            from skimage.feature import peak_local_max
        except Exception as e:
            raise ImportError("El metodo 'maximos' necesita scikit-image "
                              "(pip install scikit-image).") from e
        # coords: array de (fila, columna) = (y, x)
        coords = peak_local_max(suav, min_distance=int(min_distancia),
                                threshold_abs=t, labels=mask,
                                exclude_border=False)
        puntos = [(int(c), int(r)) for r, c in coords]          # -> (x, y)
    else:  # "centroides"
        try:
            from skimage.measure import label, regionprops
        except Exception as e:
            raise ImportError("El metodo 'centroides' necesita scikit-image "
                              "(pip install scikit-image).") from e
        for reg in regionprops(label(mask)):
            if reg.area < area_min:
                continue
            if area_max is not None and reg.area > area_max:
                continue
            r, c = reg.centroid                                  # (fila, columna)
            puntos.append((int(round(c)), int(round(r))))        # -> (x, y)

    # Recortar a los limites de la imagen por seguridad
    h, w = canal.shape
    puntos = [(min(max(x, 0), w - 1), min(max(y, 0), h - 1)) for x, y in puntos]
    return puntos


# =========================================================
# ENTRADA / SALIDA
# =========================================================
def cargar_raw(image_path):
    """Carga el TIFF tal cual (igual que el anotador: tiff.asarray())."""
    with tiff.TiffFile(str(image_path)) as tf:
        return tf.asarray()


def escribir_csv(puntos, path):
    """Escribe las posiciones como CSV de dos columnas X, Y (sin clase).

    Formato de INFERENCIA para el clasificador: sin columna 'Clase' para que
    clasificacion.py no active el modo evaluacion.
    """
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    if EXPORTAR_ENTEROS:
        xs = [int(round(x)) for x in xs]
        ys = [int(round(y)) for y in ys]
    df = pd.DataFrame({"X": xs, "Y": ys})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def ruta_csv_para(tif_path, dir_salida):
    """Deriva la ruta del CSV de salida para un TIFF dado.

    dir_salida None -> junto al TIFF; en caso contrario, en esa carpeta.
    El nombre base coincide con el del TIFF (+ SUFIJO_CSV) para que
    load_pairs_from_dir del clasificador lo empareje.
    """
    tif_path = Path(tif_path)
    nombre = f"{tif_path.stem}{SUFIJO_CSV}.csv"
    carpeta = tif_path.parent if dir_salida is None else Path(dir_salida)
    return carpeta / nombre


# =========================================================
# PROCESO POR LOTES
# =========================================================
def listar_imagenes(directorio):
    """Devuelve la lista ordenada de TIFF del directorio (sin recursividad)."""
    directorio = Path(directorio)
    if not directorio.is_dir():
        raise NotADirectoryError(f"No es un directorio: {directorio}")
    archivos = []
    for ext in EXTENSIONES:
        archivos.extend(directorio.glob(f"*{ext}"))
    # Evitar duplicados si .tif y .tiff coincidieran en algun sistema
    return sorted(set(archivos))


def procesar_directorio(directorio, dir_salida=DIRECTORIO_SALIDA,
                        sobrescribir=SOBRESCRIBIR):
    """Procesa todos los TIFF del directorio y escribe un CSV por imagen.

    Returns:
        dict con el resumen: procesadas, omitidas, con_error, total_puntos.
    """
    imagenes = listar_imagenes(directorio)
    print(f"Directorio: {Path(directorio).absolute()}")
    print(f"Imagenes encontradas: {len(imagenes)}")
    if dir_salida is not None:
        print(f"CSV de salida en: {Path(dir_salida).absolute()}")
    else:
        print("CSV de salida: junto a cada imagen (mismo nombre base).")
    print("-" * 60)

    procesadas = omitidas = con_error = total_puntos = 0

    for tif in imagenes:
        csv_out = ruta_csv_para(tif, dir_salida)

        if csv_out.exists() and not sobrescribir:
            print(f"[OMITIDA] {tif.name}: ya existe {csv_out.name} "
                  f"(usa --sobrescribir para regenerar).")
            omitidas += 1
            continue

        try:
            raw = cargar_raw(tif)
            canal = canal_estaticas(raw)
            puntos = detectar_estaticas(canal)
            escribir_csv(puntos, csv_out)
            total_puntos += len(puntos)
            procesadas += 1
            aviso = "  (0 detecciones)" if len(puntos) == 0 else ""
            print(f"[OK] {tif.name}: {len(puntos)} posiciones -> {csv_out.name}{aviso}")
        except Exception as e:
            con_error += 1
            print(f"[ERROR] {tif.name}: {e}")

    print("-" * 60)
    print(f"Resumen: {procesadas} procesadas, {omitidas} omitidas, "
          f"{con_error} con error.")
    if procesadas:
        print(f"Total de posiciones extraidas: {total_puntos} "
              f"(media {total_puntos / procesadas:.1f} por imagen).")

    return {"procesadas": procesadas, "omitidas": omitidas,
            "con_error": con_error, "total_puntos": total_puntos}


def main():
    parser = argparse.ArgumentParser(
        description="Extrae posiciones (X,Y) de trayectorias estaticas de "
                    "un directorio de TIFF de 2 canales, en CSV sin clase.")
    parser.add_argument("directorio", nargs="?", default=None,
                        help="Directorio con las imagenes TIFF. Si se omite, "
                             "se usa DIRECTORIO_IMAGENES de la configuracion.")
    parser.add_argument("--salida", default=None,
                        help="Carpeta de salida de los CSV. Por defecto, junto "
                             "a cada TIFF.")
    parser.add_argument("--sobrescribir", action="store_true",
                        help="Regenera los CSV aunque ya existan.")
    args = parser.parse_args()

    directorio = args.directorio or DIRECTORIO_IMAGENES
    dir_salida = args.salida if args.salida is not None else DIRECTORIO_SALIDA
    sobrescribir = args.sobrescribir or SOBRESCRIBIR

    if not Path(directorio).is_dir():
        print(f"[ERROR] No existe el directorio: {directorio}")
        sys.exit(1)

    procesar_directorio(directorio, dir_salida=dir_salida,
                        sobrescribir=sobrescribir)


if __name__ == "__main__":
    main()