#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anotador interactivo de partículas sobre imágenes TIFF multicanal.

Flujo:
  - Abre una imagen TIFF (channel-first o channel-last; 1, 2 o 3 canales).
  - Clic IZQUIERDO: añade un punto con la clase activa,
                    o reetiqueta el punto existente más cercano (snap).
  - Clic DERECHO:   borra el punto más cercano.
  - Teclas 1,2,3:    cambian la clase activa.
  - Tecla 'u':      deshacer la última acción.
  - Tecla 's':      guardar CSV (X, Y, Clase).
  - RadioButtons y botón "Guardar CSV" también disponibles en la barra izquierda.
  - Zoom / desplazamiento: usa la barra de herramientas de matplotlib
    (la lupa / la mano). Mientras una herramienta esté activa, los clics
    NO añaden puntos (así puedes navegar sin crear marcas por error).

Ejecútalo como SCRIPT (no como celda de notebook) para que la ventana sea
interactiva:

    python anotador_particulas.py [imagen.tif] [salida.csv]

Si no aparece ninguna ventana, te falta un backend interactivo de matplotlib
(instala tkinter:  sudo apt install python3-tk   o bien  pip install PyQt5).
"""
import sys
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button

# =========================================================
# CONFIGURACIÓN
# =========================================================
IMAGE_PATH = "SUb_02_10_merged.tif"   # imagen a anotar
# CSV de salida (X, Y, Clase).
#   - Si lo dejas en None, se DERIVA del nombre de la imagen (recomendado),
#     de modo que cada imagen tiene su propio CSV y se retoma sola.
#   - Si pones una ruta fija aquí, se usa esa y no se deriva.
OUTPUT_CSV = None

# Solo se usan cuando OUTPUT_CSV es None (CSV derivado por imagen):
ANNOT_DIR = "anotaciones"        # carpeta donde guardar/buscar los CSV por imagen
ANNOT_SUFFIX = "_anotaciones"    # imagen.tif -> anotaciones/imagen_anotaciones.csv

# Si el CSV de ESA imagen ya existe, lo carga para RETOMAR donde lo dejaste.
# Como el CSV es por imagen, una imagen nueva empieza en blanco automáticamente.
LOAD_EXISTING = True

# Opcional: CSV con posiciones X,Y (y opcionalmente Clase) generado por tu
# macro de Fiji, para PRECARGAR los puntos y limitarte a clasificarlos.
# Déjalo en None si anotas desde cero. Tiene prioridad sobre LOAD_EXISTING.
INPUT_POINTS_CSV = None

# Clases disponibles (el orden define las teclas 1, 2, 3, ...)
CLASSES = ["Borde", "Interior", "Aislada"]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # un color por clase

# Color con el que se pinta cada canal de la imagen (composición aditiva).
# ch0 -> rojo, ch1 -> verde. Ajústalo a tu caso (p.ej. ["gray", "green"]).
CHANNEL_COLORS = ["red", "green"]

# Estiramiento de contraste por percentiles (para ver señal débil).
P_LOW, P_HIGH = 1.0, 99.5

SNAP_RADIUS = 6       # px: si clicas a < SNAP_RADIUS de un punto, lo reetiquetas/borras.
                      # Bajo = más finura para partículas pegadas (ajustable con +/-).
# EXPORT_INT = True  -> X, Y como enteros (posición de píxel; recomendado, coherente
#                       con el recorte de patches y la EDT que trabajan en índices).
# EXPORT_INT = False -> conserva los decimales del clic (fracción de píxel).
EXPORT_INT = True
AUTOSAVE_ON_CLOSE = True   # guarda automáticamente al cerrar la ventana
# =========================================================

COLOR_VECS = {
    "red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1),
    "gray": (1, 1, 1), "magenta": (1, 0, 1), "cyan": (0, 1, 1),
    "yellow": (1, 1, 0),
}

CLASS_ALIASES = ("clase", "class", "label", "etiqueta")


def normalize(ch):
    """Normaliza un canal a [0, 1] con estiramiento por percentiles."""
    ch = ch.astype(float)
    lo, hi = np.percentile(ch, [P_LOW, P_HIGH])
    if hi <= lo:
        lo, hi = float(ch.min()), float(ch.max())
        if hi <= lo:
            return np.zeros_like(ch)
    return np.clip((ch - lo) / (hi - lo), 0, 1)


def to_display_rgb(image):
    """Convierte la imagen (cualquier orden/canales) a RGB float [0, 1]."""
    img = image
    if img.ndim == 2:
        g = normalize(img)
        return np.stack([g, g, g], axis=-1)

    # channel-first -> channel-last (heurística: el eje más pequeño son canales)
    if img.ndim == 3 and img.shape[0] < img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))

    h, w, c = img.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    for i in range(c):
        color_name = CHANNEL_COLORS[i] if i < len(CHANNEL_COLORS) else "gray"
        vec = COLOR_VECS.get(color_name, (1, 1, 1))
        norm = normalize(img[..., i])
        for k in range(3):
            rgb[..., k] += norm * vec[k]
    return np.clip(rgb, 0, 1)


class Anotador:
    def __init__(self, image_path, output_csv):
        # Archivo de sesión CANÓNICO: es el que se autoguarda y se carga para
        # retomar. No cambia aunque uses "Guardar como..." para exportar copias.
        self.session_csv = Path(output_csv)
        # Crear su carpeta ya, para que exista al retomar y al abrir el diálogo.
        self.session_csv.parent.mkdir(parents=True, exist_ok=True)

        # Cargar imagen
        with tiff.TiffFile(str(image_path)) as tf:
            self.raw = tf.asarray()
        self.disp = to_display_rgb(self.raw)
        self.h, self.w = self.disp.shape[:2]

        # Estado de anotación: lista de [x, y, clase]
        self.points = []
        self.history = []
        self.active_class = CLASSES[0]
        self.snap_radius = SNAP_RADIUS

        # Precarga de puntos
        if INPUT_POINTS_CSV and Path(INPUT_POINTS_CSV).exists():
            self._load_csv(Path(INPUT_POINTS_CSV))
        elif LOAD_EXISTING and self.session_csv.exists():
            self._load_csv(self.session_csv)
            print(f"[INFO] Retomando anotación previa de esta imagen ({self.session_csv}).")
        else:
            print(f"[INFO] Imagen nueva: empiezo en blanco. Se guardará en {self.session_csv}.")

        self._build_ui()
        self._redraw()

    # ---------- carga / guardado ----------
    def _load_csv(self, path):
        df = pd.read_csv(path)
        class_col = None
        for col in df.columns:
            if col.strip().lower() in CLASS_ALIASES:
                class_col = col
                break
        for _, row in df.iterrows():
            if class_col and not pd.isna(row[class_col]):
                clase = str(row[class_col])
            else:
                clase = CLASSES[0]
            self.points.append([float(row["X"]), float(row["Y"]), clase])
        print(f"[INFO] Cargados {len(self.points)} puntos de {path}")

    def _write_csv(self, path):
        """Escribe los puntos actuales en 'path' como X, Y, Clase."""
        if not self.points:
            print("[INFO] No hay puntos que guardar.")
            return False
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        if EXPORT_INT:
            xs = [int(round(x)) for x in xs]
            ys = [int(round(y)) for y in ys]
        df = pd.DataFrame({"X": xs, "Y": ys, "Clase": [p[2] for p in self.points]})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[OK] Guardado {len(df)} puntos en {path}")
        return True

    def save(self, *_):
        """Guardado rápido en el archivo de sesión canónico (sin diálogo).

        Es el archivo que se carga al retomar, así que siempre va al sitio
        correcto. Tecla 's' y botón 'Guardar CSV'.
        """
        self._write_csv(self.session_csv)

    def save_as(self, *_):
        """Exporta una COPIA a la ruta que elijas en un diálogo.

        No cambia el archivo de sesión: el autoguardado y el retomar siguen
        usando self.session_csv.
        """
        if not self.points:
            print("[INFO] No hay puntos que guardar.")
            return
        path = self._ask_save_path()
        if path:                       # None si el usuario cancela
            self._write_csv(path)

    def _ask_save_path(self):
        """Diálogo 'Guardar como'. Devuelve la ruta elegida o None si se cancela."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()            # ocultar la ventana raíz
            root.attributes("-topmost", True)
            path = filedialog.asksaveasfilename(
                title="Exportar copia de anotaciones como...",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
                initialdir=str(self.session_csv.parent),
                initialfile=self.session_csv.name,
            )
            root.destroy()
            return path if path else None
        except Exception as e:
            # Si tkinter no está disponible, caemos al archivo de sesión.
            print(f"[AVISO] No se pudo abrir el diálogo ({e}); "
                  f"guardo en {self.session_csv}")
            return self.session_csv

    # ---------- interfaz ----------
    def _build_ui(self):
        self.fig = plt.figure(figsize=(13, 8))
        self.ax_img = self.fig.add_axes([0.22, 0.06, 0.76, 0.9])
        self.ax_img.imshow(self.disp, origin="upper")
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        self.scatter = self.ax_img.scatter(
            [], [], s=42, edgecolors="white", linewidths=0.9, zorder=3
        )

        # RadioButtons de clases (ya van con su color, hacen de leyenda)
        ax_radio = self.fig.add_axes([0.02, 0.58, 0.16, 0.30])
        ax_radio.set_title("Clase activa", fontsize=10)
        self.radio = RadioButtons(ax_radio, CLASSES, active=0)
        for lbl, col in zip(self.radio.labels, CLASS_COLORS):
            lbl.set_color(col)
        self.radio.on_clicked(self._on_radio)

        # Botón: guardado rápido en el archivo de sesión (lo que se retoma)
        ax_save = self.fig.add_axes([0.02, 0.49, 0.16, 0.055])
        self.btn_save = Button(ax_save, "Guardar CSV")
        self.btn_save.on_clicked(self.save)

        # Botón: exportar una copia a otra ruta (no afecta al retomar)
        ax_saveas = self.fig.add_axes([0.02, 0.42, 0.16, 0.055])
        self.btn_saveas = Button(ax_saveas, "Guardar como…")
        self.btn_saveas.on_clicked(self.save_as)

        # Instrucciones
        self.fig.text(
            0.02, 0.34,
            "Izq: añadir / reetiquetar\n"
            "Der: borrar más cercano\n"
            "Teclas 1-3: clase\n"
            "u: deshacer   s: guardar\n"
            "+ / - : radio de selección\n"
            "Lupa/mano: zoom (clics no\n"
            "añaden con herramienta activa)\n"
            "Zoom para partículas pegadas.",
            fontsize=9, va="top",
        )

        # Eventos
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _toolbar_active(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(tb, "mode", "")) if tb is not None else False

    # ---------- lógica ----------
    def _push_history(self):
        self.history.append(copy.deepcopy(self.points))
        if len(self.history) > 200:
            self.history.pop(0)

    def _nearest(self, x, y):
        if not self.points:
            return None, None
        d2 = [(px - x) ** 2 + (py - y) ** 2 for px, py, _ in self.points]
        i = int(np.argmin(d2))
        return i, d2[i] ** 0.5

    def _on_click(self, event):
        if event.inaxes != self.ax_img or self._toolbar_active():
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if event.button == 1:      # izquierdo: añadir o reetiquetar
            i, dist = self._nearest(x, y)
            self._push_history()
            if i is not None and dist <= self.snap_radius:
                self.points[i][2] = self.active_class
            else:
                self.points.append([x, y, self.active_class])
            self._redraw()
        elif event.button == 3:    # derecho: borrar
            i, dist = self._nearest(x, y)
            if i is not None and dist <= self.snap_radius:
                self._push_history()
                self.points.pop(i)
                self._redraw()

    def _on_key(self, event):
        if event.key in [str(n) for n in range(1, len(CLASSES) + 1)]:
            idx = int(event.key) - 1
            self.active_class = CLASSES[idx]
            self.radio.set_active(idx)   # sincroniza el RadioButton
        elif event.key == "u":
            if self.history:
                self.points = self.history.pop()
                self._redraw()
        elif event.key == "s":
            self.save()
        elif event.key in ("+", "="):    # subir radio de selección
            self.snap_radius = min(self.snap_radius + 2, 60)
            print(f"[INFO] Radio de selección: {self.snap_radius} px")
            self._redraw()
        elif event.key in ("-", "_"):    # bajar radio de selección
            self.snap_radius = max(self.snap_radius - 2, 1)
            print(f"[INFO] Radio de selección: {self.snap_radius} px")
            self._redraw()

    def _on_radio(self, label):
        self.active_class = label

    def _on_close(self, event):
        if AUTOSAVE_ON_CLOSE:
            self._write_csv(self.session_csv)

    def _redraw(self):
        if self.points:
            offsets = np.array([[p[0], p[1]] for p in self.points])
            colors = [
                CLASS_COLORS[CLASSES.index(p[2])] if p[2] in CLASSES else "white"
                for p in self.points
            ]
            self.scatter.set_offsets(offsets)
            self.scatter.set_facecolor(colors)
        else:
            self.scatter.set_offsets(np.empty((0, 2)))
            self.scatter.set_facecolor([])

        counts = {cl: sum(1 for p in self.points if p[2] == cl) for cl in CLASSES}
        resumen = "   ".join(f"{cl}: {counts[cl]}" for cl in CLASSES)
        self.ax_img.set_title(
            f"Clase activa: {self.active_class}   |   {resumen}   |   "
            f"total: {len(self.points)}   |   radio: {self.snap_radius}px"
        )
        self.fig.canvas.draw_idle()


def derive_csv_path(image_path):
    """Deriva la ruta del CSV a partir del nombre de la imagen.

    ejemplo:  fotos/imagen2.tif  ->  anotaciones/imagen2_anotaciones.csv
    """
    stem = Path(image_path).stem
    return str(Path(ANNOT_DIR) / f"{stem}{ANNOT_SUFFIX}.csv")


def main():
    parser = argparse.ArgumentParser(description="Anotador de partículas TIFF -> CSV")
    parser.add_argument("image", nargs="?", default=IMAGE_PATH)
    parser.add_argument("output", nargs="?", default=None,
                        help="CSV de salida. Si se omite, se deriva del nombre de la imagen.")
    args = parser.parse_args()

    if not Path(args.image).exists():
        sys.exit(f"[ERROR] No existe la imagen: {args.image}")

    # Prioridad: ruta por línea de comandos > OUTPUT_CSV fijo > derivado por imagen
    output = args.output or OUTPUT_CSV
    if output is None:
        output = derive_csv_path(args.image)

    Anotador(args.image, output)
    plt.show()


if __name__ == "__main__":
    main()