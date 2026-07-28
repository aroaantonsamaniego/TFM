import sys
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import RadioButtons, Button, Slider

# =========================================================
# CONFIGURACION
# =========================================================
IMAGE_PATH = "../datos/imagenes_nuevas/SUboligo_02_3.tif"   # imagen a anotar
# CSV de salida (X, Y, Clase).
#   - None  -> se DERIVA del nombre de la imagen (recomendado): cada imagen
#              tiene su propio CSV y se retoma sola.
#   - ruta  -> se usa esa y no se deriva.
OUTPUT_CSV = None

# Solo se usan cuando OUTPUT_CSV es None (CSV derivado por imagen):
ANNOT_DIR = "../datos/definitivos"        # carpeta donde guardar/buscar los CSV por imagen
#ANNOT_SUFFIX = "_anotaciones"    # imagen.tif -> anotaciones/imagen_anotaciones.csv este archivo es el que retoma
ANNOT_SUFFIX = ""

# Si el CSV de ESA imagen ya existe, lo carga para RETOMAR donde lo dejaste
# (incluyendo quE puntos siguen sin clasificar). Una imagen nueva se detecta
# automáticamente desde cero.
LOAD_EXISTING = True

# Opcional: CSV con posiciones X,Y (y opcionalmente Clase) ya generado (p.ej.
# por una macro de Fiji). Si se indica, PRECARGA esos puntos en lugar de
# detectarlos. Déjalo en None para usar la detección automática.
INPUT_POINTS_CSV = None

# ── Detección automática de trayectorias estaticas (canal rojo, canal 0) ─────
DETECTAR_AUTOMATICO = True       # al abrir una imagen nueva, detecta y precarga
METODO_DETECCION    = "maximos"  # "maximos" (picos locales) | "centroides" (blobs)
DET_SIGMA           = 1.0        # suavizado gaussiano previo (px); 0 = sin suavizar
DET_UMBRAL          = None       # None -> Otsu automático sobre el canal rojo
DET_MIN_DISTANCIA   = 3         # ("maximos")   separación mínima entre picos (px)
DET_AREA_MIN        = 2          # ("centroides") área mínima de un blob (px)
DET_AREA_MAX        = None       # ("centroides") área máxima (None = sin límite)

# Permitir anyadir puntos a mano (clic izq. en zona vacía). En False, un clic
# fuera de un punto no hace nada: solo se pueden clasificar los detectados.
PERMITIR_ANADIR = False

# Clases disponibles (el orden define las teclas 1, 2, 3, ...)
CLASSES = ["Borde", "Interior", "Aislada"]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # un color por clase
COLOR_SIN_CLASIFICAR = "#dddddd"  # contorno de los puntos aún sin clasificar

# Color con el que se pinta cada canal de la imagen (composición aditiva).
# ch0 -> rojo (estáticas), ch1 -> verde (registradas).
CHANNEL_COLORS = ["red", "green"]

# Estiramiento de contraste por percentiles (para ver señal débil).
P_LOW, P_HIGH = 1.0, 99.5

SNAP_RADIUS = 6       # px: si clicas a < SNAP_RADIUS de un punto, lo reetiquetas/borras.
EXPORT_INT = True     # X, Y como enteros (coherente con el recorte de patches y la EDT).
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


# =========================================================
# DETECCION AUTOMATICA DE TRAYECTORIAS ESTATICAS
# =========================================================
def canal_estaticas(image):
    """Devuelve el canal de estáticas (canal 0) como 2D float [0, 1].

    Replica la convención de load_tif_image: el canal 0 es el rojo (estáticas).
    Acepta (H,W), (2,H,W)/(C,H,W) channel-first o (H,W,C) channel-last.
    """
    img = image
    if img.ndim == 2:
        ch = img
    else:
        # channel-first -> channel-last si el primer eje es el mas pequeño
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
                                           Un punto por maximo; separa estaticas
                                           próximas con min_distancia.
                            "centroides" -> umbral + componentes conexas; un punto
                                           por blob (centroide), con filtro de área.
        umbral       (float|None): Umbral de intensidad. None -> Otsu automático.
        sigma        (float): Suavizado gaussiano previo (px). 0 = sin suavizar.
        min_distancia(int):   ("maximos") separación mínima entre picos (px).
        area_min     (int):   ("centroides") área mínima de un blob (px).
        area_max     (int|None): ("centroides") área máxima (None = sin límite).

    Returns:
        list[(x, y)]: coordenadas enteras (columna, fila) de cada estática.
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
            raise ImportError("El método 'maximos' necesita scikit-image "
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
            raise ImportError("El método 'centroides' necesita scikit-image "
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


class Anotador:
    def __init__(self, image_path, output_csv):
        # Archivo de sesion caonico: el que se autoguarda y se carga al retomar.
        self.session_csv = Path(output_csv)
        self.session_csv.parent.mkdir(parents=True, exist_ok=True)

        # Cargar imagen
        with tiff.TiffFile(str(image_path)) as tf:
            self.raw = tf.asarray()
        # Canales normalizados por separado + sus vectores de color, para poder
        # recomponer la imagen al mover los controles de contraste.
        self.base_channels, self.channel_vecs = self._prep_channels(self.raw)
        self.n_ch = len(self.base_channels)
        self.gains = [1.0] * self.n_ch            # ganancia de contraste por canal
        self.h, self.w = self.base_channels[0].shape
        self.disp = self._compose_rgb()

        # Estado de anotacion: lista de [x, y, clase]. clase = None -> sin clasificar.
        self.points = []
        self.history = []
        self.active_class = CLASSES[0]
        self.snap_radius = SNAP_RADIUS

        # ── Precarga / deteccion ─────────────────────────────────────────────
        precargado = False
        if INPUT_POINTS_CSV and Path(INPUT_POINTS_CSV).exists():
            self._load_csv(Path(INPUT_POINTS_CSV))
            precargado = True
        elif LOAD_EXISTING and self.session_csv.exists():
            self._load_csv(self.session_csv)
            print(f"[INFO] Retomando anotación previa de esta imagen "
                  f"({self.session_csv}).")
            precargado = True

        if not precargado:
            if DETECTAR_AUTOMATICO:
                canal = canal_estaticas(self.raw)
                detectadas = detectar_estaticas(canal)
                self.points = [[float(x), float(y), None] for x, y in detectadas]
                print(f"[INFO] Detectadas {len(detectadas)} trayectorias estáticas "
                      f"(método '{METODO_DETECCION}'). Todas SIN CLASIFICAR.")
                # Registrarlas ya en el CSV de sesión.
                self._write_csv(self.session_csv)
            else:
                print(f"[INFO] Imagen nueva sin detección automática: empiezo en "
                      f"blanco. Se guardará en {self.session_csv}.")

        self._build_ui()
        self._redraw()

    # ---------- composicion de la imagen / contraste ----------
    def _prep_channels(self, image):
        """Normaliza cada canal a [0,1] y devuelve (bases, vectores_color)."""
        img = image
        if img.ndim == 2:
            return [normalize(img)], [COLOR_VECS.get("gray", (1, 1, 1))]
        if img.ndim == 3 and img.shape[0] < img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))   # channel-first -> channel-last
        c = img.shape[-1]
        bases, vecs = [], []
        for i in range(c):
            color_name = CHANNEL_COLORS[i] if i < len(CHANNEL_COLORS) else "gray"
            bases.append(normalize(img[..., i]))
            vecs.append(COLOR_VECS.get(color_name, (1, 1, 1)))
        return bases, vecs

    def _compose_rgb(self):
        """Recompone la imagen RGB aplicando la ganancia de contraste por canal."""
        rgb = np.zeros((self.h, self.w, 3), dtype=float)
        for base, vec, g in zip(self.base_channels, self.channel_vecs, self.gains):
            norm = np.clip(base * g, 0.0, 1.0)
            for k in range(3):
                rgb[..., k] += norm * vec[k]
        return np.clip(rgb, 0.0, 1.0)

    def _on_contrast(self, idx, val):
        """Callback de las barras de contraste: recompone y redibuja."""
        self.gains[idx] = float(val)
        self.im.set_data(self._compose_rgb())
        self.fig.canvas.draw_idle()

    # ---------- carga / guardado ----------
    def _match_clase(self, valor):
        """Devuelve la clase canónica (de CLASSES) o None si no clasifica."""
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            return None
        c = str(valor).strip()
        if not c:
            return None
        for k in CLASSES:
            if k.lower() == c.lower():
                return k
        return None   # etiqueta no reconocida -> se trata como sin clasificar

    def _load_csv(self, path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        class_col = None
        for col in df.columns:
            if col.strip().lower() in CLASS_ALIASES:
                class_col = col
                break
        for _, row in df.iterrows():
            clase = self._match_clase(row[class_col]) if class_col else None
            self.points.append([float(row["X"]), float(row["Y"]), clase])
        n_sc = sum(1 for p in self.points if p[2] is None)
        print(f"[INFO] Cargados {len(self.points)} puntos de {path} "
              f"({n_sc} sin clasificar).")

    def _write_csv(self, path):
        """Escribe los puntos actuales como X, Y, Clase (vacío = sin clasificar)."""
        if not self.points:
            print("[INFO] No hay puntos que guardar.")
            return False
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        if EXPORT_INT:
            xs = [int(round(x)) for x in xs]
            ys = [int(round(y)) for y in ys]
        clases = [(p[2] if p[2] is not None else "") for p in self.points]
        df = pd.DataFrame({"X": xs, "Y": ys, "Clase": clases})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        n_sc = sum(1 for c in clases if c == "")
        msg = f"[OK] Guardado {len(df)} puntos en {path}"
        if n_sc:
            msg += f"  (¡{n_sc} aún sin clasificar!)"
        print(msg)
        return True

    def save(self, *_):
        """Guardado rápido en el archivo de sesión canónico (tecla 's' / botón)."""
        self._write_csv(self.session_csv)

    def save_as(self, *_):
        """Exporta una COPIA a la ruta que elijas (no cambia el archivo de sesión)."""
        if not self.points:
            print("[INFO] No hay puntos que guardar.")
            return
        path = self._ask_save_path()
        if path:
            self._write_csv(path)

    def _ask_save_path(self):
        """Diálogo 'Guardar como'. Devuelve la ruta elegida o None si se cancela."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
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
            print(f"[AVISO] No se pudo abrir el diálogo ({e}); "
                  f"guardo en {self.session_csv}")
            return self.session_csv

    # ---------- interfaz ----------
    def _build_ui(self):
        self.fig = plt.figure(figsize=(13, 8))
        self.ax_img = self.fig.add_axes([0.22, 0.11, 0.76, 0.85])
        self.im = self.ax_img.imshow(self.disp, origin="upper")
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        self.scatter = self.ax_img.scatter(
            [], [], s=46, edgecolors="white", linewidths=1.3, zorder=3
        )

        # RadioButtons de clases (hacen también de leyenda con su color)
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

        # Controles de contraste: una barra por canal (rojo y verde) bajo la imagen
        self.sliders = []
        nombres = ["Contraste rojo", "Contraste verde", "Contraste azul"]
        colores = ["#c0392b", "#1e8449", "#2471a3"]
        xs = [0.30, 0.66, 0.30]
        for i in range(min(self.n_ch, 2)):
            ax_s = self.fig.add_axes([xs[i], 0.045, 0.25, 0.025])
            s = Slider(ax_s,
                       nombres[i] if i < len(nombres) else f"Ch{i}",
                       0.2, 6.0, valinit=self.gains[i])
            s.label.set_color(colores[i] if i < len(colores) else "black")
            s.label.set_fontsize(9)
            s.on_changed(lambda val, idx=i: self._on_contrast(idx, val))
            self.sliders.append(s)

        # Instrucciones
        self.fig.text(
            0.02, 0.36,
            "Hueco = sin clasificar\n"
            "Relleno = ya anotado\n"
            "Izq sobre punto: asignar clase\n"
            "Der: quitar etiqueta (no borra)\n"
            "x / Supr: borrar punto\n"
            "Teclas 1-3: clase   d: desclasificar\n"
            "u: deshacer   s: guardar\n"
            "+ / - : radio de selección\n"
            "Barras inferiores: contraste R/G\n"
            "Lupa/mano: zoom",
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

        if event.button == 1:      # izquierdo: clasificar punto o añadir
            i, dist = self._nearest(x, y)
            if i is not None and dist <= self.snap_radius:
                self._push_history()
                self.points[i][2] = self.active_class   # hueco -> relleno (o reetiqueta)
                self._redraw()
            elif PERMITIR_ANADIR:
                self._push_history()
                self.points.append([x, y, self.active_class])
                self._redraw()
        elif event.button == 3:    # derecho: quitar etiqueta (mantiene el redondel)
            i, dist = self._nearest(x, y)
            if (i is not None and dist <= self.snap_radius
                    and self.points[i][2] is not None):
                self._push_history()
                self.points[i][2] = None
                self._redraw()

    def _on_key(self, event):
        if event.key in [str(n) for n in range(1, len(CLASSES) + 1)]:
            idx = int(event.key) - 1
            self.active_class = CLASSES[idx]
            self.radio.set_active(idx)   # sincroniza el RadioButton
        elif event.key == "d":           # desclasificar el punto bajo el cursor
            if event.xdata is not None and event.ydata is not None:
                i, dist = self._nearest(float(event.xdata), float(event.ydata))
                if i is not None and dist <= self.snap_radius:
                    self._push_history()
                    self.points[i][2] = None
                    self._redraw()
        elif event.key in ("x", "delete", "backspace"):  # borrar punto bajo cursor
            if event.xdata is not None and event.ydata is not None:
                i, dist = self._nearest(float(event.xdata), float(event.ydata))
                if i is not None and dist <= self.snap_radius:
                    self._push_history()
                    self.points.pop(i)
                    self._redraw()
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
            faces, edges = [], []
            for p in self.points:
                if p[2] in CLASSES:
                    faces.append(mcolors.to_rgba(CLASS_COLORS[CLASSES.index(p[2])]))
                    edges.append(mcolors.to_rgba("white"))
                else:                                   # sin clasificar -> hueco
                    faces.append((0.0, 0.0, 0.0, 0.0))
                    edges.append(mcolors.to_rgba(COLOR_SIN_CLASIFICAR))
            self.scatter.set_offsets(offsets)
            self.scatter.set_facecolor(faces)
            self.scatter.set_edgecolor(edges)
        else:
            self.scatter.set_offsets(np.empty((0, 2)))
            self.scatter.set_facecolor([])
            self.scatter.set_edgecolor([])

        sin_clasificar = sum(1 for p in self.points if p[2] not in CLASSES)
        counts = {cl: sum(1 for p in self.points if p[2] == cl) for cl in CLASSES}
        resumen = "   ".join(f"{cl}: {counts[cl]}" for cl in CLASSES)
        self.ax_img.set_title(
            f"Clase activa: {self.active_class}   |   Sin clasificar: {sin_clasificar}"
            f"   |   {resumen}   |   total: {len(self.points)}   |   "
            f"radio: {self.snap_radius}px"
        )
        self.fig.canvas.draw_idle()


def derive_csv_path(image_path):
    """Deriva la ruta del CSV a partir del nombre de la imagen.

    ejemplo:  fotos/imagen2.tif  ->  anotaciones/imagen2_anotaciones.csv
    """
    stem = Path(image_path).stem
    return str(Path(ANNOT_DIR) / f"{stem}{ANNOT_SUFFIX}.csv")


def pedir_imagen():
    """Abre un diálogo gráfico para elegir la imagen TIFF a anotar.

    Devuelve la ruta elegida (str) o None si se cancela o si Tkinter no está
    disponible.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Elige la imagen TIFF a anotar",
            filetypes=[("Imágenes TIFF", "*.tif *.tiff"), ("Todos", "*.*")],
        )
        root.destroy()
        return path or None
    except Exception as e:
        print(f"[AVISO] No se pudo abrir el selector de archivo: {e}")
        return None


def _mostrar_error(msg):
    """Muestra un error por ventana (modo sin consola) y por consola si la hay."""
    print(f"[ERROR] {msg}")
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror("Anotador", msg)
        root.destroy()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Anotador de trayectorias TIFF -> CSV")
    parser.add_argument("image", nargs="?", default=None,
                        help="Imagen TIFF a anotar. Si se omite, se abre un selector.")
    parser.add_argument("output", nargs="?", default=None,
                        help="CSV de salida. Si se omite, se deriva del nombre de la imagen.")
    args = parser.parse_args()

    # Imagen: línea de comandos > selector gráfico > IMAGE_PATH (modo desarrollo)
    image = args.image
    if image is None:
        image = pedir_imagen()
        if not image and Path(IMAGE_PATH).exists():
            image = IMAGE_PATH

    if not image:
        _mostrar_error("No se eligió ninguna imagen. Cierro el programa.")
        sys.exit(1)
    if not Path(image).exists():
        _mostrar_error(f"No existe la imagen:\n{image}")
        sys.exit(1)

    # Prioridad CSV: ruta por línea de comandos > OUTPUT_CSV fijo > derivado por imagen
    output = args.output or OUTPUT_CSV
    if output is None:
        output = derive_csv_path(image)

    Anotador(image, output)
    plt.show()


if __name__ == "__main__":
    main()
