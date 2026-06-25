#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detectar_aisladas.py
================================================================================
Detecta partículas AISLADAS: la EDT aplicada al canal
verde (distancia a las trayectorias).

  Criterio (verdad de campo): una partícula es "aislada" si está FUERA de las
  trazas verdes (ni en su interior ni en su borde).

  Cómo se aplica la EDT aquí:
    1. Mg  = canal_verde > t_g            (binarización del verde; Otsu si None).
    2. D_g = distance_transform_edt(~Mg)  (distancia de CADA píxel a la traza
                                           verde más cercana; el verde es el 0).
    3. Para cada partícula del CSV se lee d_g = D_g[y, x].
    4. aislada ⟺ d_g > θ  (umbral_dist). Dentro de la banda (d_g ≤ θ) = borde o
       interior = NO aislada.
    (Opcional) Conectividad: si la isla roja de la partícula toca una traza verde
    aunque su centroide quede algo lejos, se considera NO aislada.

Se evalúa contra la columna 'clase' del CSV, donde 'aislada'/'aislado' es la
clase positiva. Por el fuerte desbalanceo, mira F1 y kappa, no solo accuracy.

────────────────────────────────────────────────────────────────────────────────
USO — archivo individual:
    # Evaluar el método con parámetros fijos (requiere columna 'clase')
    python detectar_aisladas.py --tif img.tif --csv datos.csv
    # Barrido de parámetros del método (umbral del verde × umbral de distancia)
    python detectar_aisladas.py --tif img.tif --csv datos.csv --barrer
    # Inferencia (CSV sin columna 'clase'): clasifica y guarda, sin métricas
    python detectar_aisladas.py --tif img.tif --csv datos_sin_clase.csv --guardar-csv out.csv
    # Filtrado -> genera <base>_filtrado.csv con columnas X, Y, clase, Filtradas
    python detectar_aisladas.py --tif img.tif --csv datos.csv --filtrar

USO — directorio completo (empareja cada .tif con su .csv de igual nombre base):
    python detectar_aisladas.py --dir /ruta/
    python detectar_aisladas.py --dir /ruta/ --barrer       # barrido GLOBAL acumulado
    python detectar_aisladas.py --dir /ruta/ --filtrar      # un _filtrado.csv por par

Rejillas de barrido personalizables:
    python detectar_aisladas.py --dir /ruta/ --barrer \
        --umbrales-verde 0.01 0.05 0.1 --umbrales-dist 0 1 2 3 5 8
================================================================================
"""

import matplotlib
matplotlib.use("Agg")          # backend sin display, para servidor remoto (hertz)

import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, apply_hysteresis_threshold

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.patches as mpatches

# ── Función del pipeline reutilizada ──────────────────────────────────────────
# El módulo auxiliar debe estar en el mismo directorio que este script (o en el
# PYTHONPATH). Si cambias de versión del módulo, edita SOLO esta línea.
from funciones_auxiliares import load_tif_image


# ──────────────────────────────────────────────────────────────────────────────
# Etiquetas consideradas en la verdad de campo
# ──────────────────────────────────────────────────────────────────────────────

CLASES_AISLADAS = {'aislada', 'aislado'}
CLASES_INTERIOR = {'interior'}
CLASES_BORDE    = {'borde'}


# ──────────────────────────────────────────────────────────────────────────────
# Rejillas de barrido por defecto
# ──────────────────────────────────────────────────────────────────────────────

REJILLA_UMBRAL_VERDE = [None]                 # None = Otsu; ampliable por CLI
REJILLA_UMBRAL_DIST  = [0, 1, 2, 3, 5, 8, 12] # umbral de distancia θ (px)


# ──────────────────────────────────────────────────────────────────────────────
# Verdad de campo desde el CSV
# ──────────────────────────────────────────────────────────────────────────────

def cargar_ground_truth(csv_path):
    '''
    Lee el CSV. Soporta dos formatos:
      - Con clase (x, y, clase): modo evaluación/barrido.
      - Sin clase (x, y):        modo inferencia.

    Args:
        csv_path (str): Ruta al CSV.

    Returns:
        tuple: (positions, clases_raw, y_true)
               - positions  : lista de (y, x) enteros (formato del pipeline).
               - clases_raw : lista de str con la clase original, o None si no hay.
               - y_true     : np.ndarray bool (True = aislada), o None si no hay clase.
    '''
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = [c.strip().lower() for c in df.columns]

    if not {'x', 'y'}.issubset(df.columns):
        raise ValueError("El CSV debe contener al menos las columnas: x, y")

    positions = list(zip(pd.to_numeric(df['y']).astype(int),
                         pd.to_numeric(df['x']).astype(int)))

    if 'clase' not in df.columns:
        return positions, None, None

    df['clase'] = df['clase'].str.strip().str.lower()
    clases_raw  = df['clase'].tolist()
    y_true      = np.array([c in CLASES_AISLADAS for c in clases_raw], dtype=bool)
    return positions, clases_raw, y_true


# ──────────────────────────────────────────────────────────────────────────────
# Criterio EDT al verde — descompuesto para reuso eficiente en el barrido
# ──────────────────────────────────────────────────────────────────────────────

def _umbral_auto(canal, umbral):
    '''Devuelve umbral fijo si se da, o el de Otsu (o la media si Otsu falla).'''
    if umbral is not None:
        return float(umbral)
    try:
        return float(threshold_otsu(canal))
    except Exception:
        return float(canal.mean())


def construir_mascara_verde(canal_verde, umbral_verde=None, umbral_verde_bajo=None,
                            verde_sigma=0.0, verde_cierre=0, verde_rellenar=False):
    '''
    Construye la máscara binaria del verde Mg de forma robusta para fluorescencia
    difusa (núcleos brillantes + halos tenues). Pasos (todos opcionales salvo el
    umbral):

      1. Suavizado gaussiano (verde_sigma): une el verde difuso antes de umbralar.
      2. Umbralización:
           - Simple:    Mg = verde > t_alto   (t_alto = umbral_verde u Otsu).
           - Histéresis (si umbral_verde_bajo): conserva los píxeles por encima de
             umbral_verde_bajo que estén CONECTADOS a algún píxel por encima de
             t_alto. Así crece desde los núcleos brillantes hacia sus halos tenues
             sin recoger ruido de fondo aislado. Es la mejor opción para "cubrir
             todo el verde" sin ensuciar.
      3. Cierre morfológico (verde_cierre): rellena huecos pequeños y conecta.
      4. Relleno de huecos (verde_rellenar): tapa los agujeros interiores de Mg.

    Returns:
        tuple: (Mg, t_alto)
    '''
    img = ndi.gaussian_filter(canal_verde, verde_sigma) \
        if (verde_sigma and verde_sigma > 0) else canal_verde

    t_alto = _umbral_auto(img, umbral_verde)

    if umbral_verde_bajo is not None:
        Mg = apply_hysteresis_threshold(img, float(umbral_verde_bajo), t_alto)
    else:
        Mg = img > t_alto

    if verde_cierre and verde_cierre > 0:
        Mg = ndi.binary_closing(Mg, iterations=int(verde_cierre))
    if verde_rellenar:
        Mg = ndi.binary_fill_holes(Mg)

    return np.asarray(Mg, dtype=bool), float(t_alto)


def _verde_base(canal_rojo, canal_verde, umbral_verde, umbral_rojo,
                usar_conectividad, umbral_verde_bajo=None, verde_sigma=0.0,
                verde_cierre=0, verde_rellenar=False):
    Mg, t_g = construir_mascara_verde(
        canal_verde, umbral_verde, umbral_verde_bajo,
        verde_sigma, verde_cierre, verde_rellenar)

    D_g = ndi.distance_transform_edt(~Mg)

    if usar_conectividad:
        t_r = _umbral_auto(canal_rojo, umbral_rojo)
        Mr = canal_rojo > t_r
        labels, _ = ndi.label(Mr | Mg)
        labels_con_verde = set(np.unique(labels[Mg]).tolist())
        labels_con_verde.discard(0)
        
        # NUEVO: Aislar las manchas rojas para ver si se salen del verde
        labels_rojo, _ = ndi.label(Mr)
        rojo_fuera = set(np.unique(labels_rojo[~Mg]).tolist())
        rojo_fuera.discard(0)
    else:
        labels = None
        labels_con_verde = set()
        labels_rojo = None
        rojo_fuera = set()

    return Mg, D_g, labels, labels_con_verde, t_g, labels_rojo, rojo_fuera


def _clasificar_verde_desde_campo(positions, D_g, labels, labels_con_verde,
                                  labels_rojo, rojo_fuera,
                                  umbral_dist, usar_conectividad, shape):
    H, W = shape
    n = len(positions)
    mask_aisladas = np.zeros(n, dtype=bool)
    mask_interior = np.zeros(n, dtype=bool)
    mask_borde    = np.zeros(n, dtype=bool)
    en_verde      = np.zeros(n, dtype=bool)
    dvals         = np.zeros(n, dtype=float)

    for i, (y, x) in enumerate(positions):
        yy = min(max(int(y), 0), H - 1)
        xx = min(max(int(x), 0), W - 1)
        d = float(D_g[yy, xx])
        dvals[i] = d

        if d <= umbral_dist:
            en_verde[i]      = True       # borde o interior
            mask_aisladas[i] = False
            
            # NUEVO: Sub-clasificación estricta (toda la mancha)
            if usar_conectividad and labels_rojo is not None:
                lbl_r = labels_rojo[yy, xx]
                if lbl_r != 0:
                    # Si la mancha roja NO tiene píxeles fuera del verde -> interior
                    if lbl_r not in rojo_fuera:
                        mask_interior[i] = True
                    else:
                        mask_borde[i] = True
                else:
                    # Fallback si el centroide no cae en la máscara roja detectada
                    if d == 0:
                        mask_interior[i] = True
                    else:
                        mask_borde[i] = True
            else:
                # Comportamiento original si no usamos conectividad
                if d == 0:
                    mask_interior[i] = True
                else:
                    mask_borde[i] = True
            continue

        if usar_conectividad:
            lbl = labels[yy, xx]
            toca_verde = (lbl != 0 and lbl in labels_con_verde)
            mask_aisladas[i] = not toca_verde
            
            if toca_verde:
                mask_borde[i] = True
        else:
            mask_aisladas[i] = True       # fuera de la banda de θ

    return mask_aisladas, mask_interior, mask_borde, en_verde, dvals


def clasificar_aisladas_verde(canal_rojo, canal_verde, positions,
                              umbral_verde=None, umbral_rojo=None,
                              umbral_dist=2.0, usar_conectividad=True,
                              umbral_verde_bajo=None, verde_sigma=0.0,
                              verde_cierre=0, verde_rellenar=False):
    Mg, D_g, labels, labels_con_verde, t_g, labels_rojo, rojo_fuera = _verde_base(
        canal_rojo, canal_verde, umbral_verde, umbral_rojo, usar_conectividad,
        umbral_verde_bajo, verde_sigma, verde_cierre, verde_rellenar)

    mask_aisladas, mask_interior, mask_borde, en_verde, dvals = _clasificar_verde_desde_campo(
        positions, D_g, labels, labels_con_verde, labels_rojo, rojo_fuera,
        umbral_dist, usar_conectividad, canal_verde.shape)

    info = {
        'dist_verde':   dvals,
        'en_verde':     en_verde,
        'interior':     mask_interior,
        'borde':        mask_borde,
        'umbral_verde': t_g,
        'umbral_dist':  float(umbral_dist),
    }
    return mask_aisladas, info

 
# ──────────────────────────────────────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────────────────────────────────────

def confusion(y_pred, y_true):
    '''Devuelve {TP, TN, FP, FN} tomando la clase target como positiva.'''
    y_pred = np.asarray(y_pred, dtype=bool)
    y_true = np.asarray(y_true, dtype=bool)
    return {
        'TP': int(np.sum( y_pred &  y_true)),
        'TN': int(np.sum(~y_pred & ~y_true)),
        'FP': int(np.sum( y_pred & ~y_true)),
        'FN': int(np.sum(~y_pred &  y_true)),
    }


def metricas_desde_confusion(c):
    '''Calcula accuracy, precision, recall, F1 y kappa a partir de {TP,TN,FP,FN}.'''
    TP, TN, FP, FN = c['TP'], c['TN'], c['FP'], c['FN']
    total    = TP + TN + FP + FN
    aciertos = TP + TN

    accuracy  = aciertos / total              if total       else 0.0
    precision = TP / (TP + FP)                if (TP + FP)   else 0.0
    recall    = TP / (TP + FN)                if (TP + FN)   else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    po     = accuracy
    p_real = (TP + FN) / total if total else 0.0
    p_pred = (TP + FP) / total if total else 0.0
    pe     = p_real * p_pred + (1 - p_real) * (1 - p_pred)
    kappa  = (po - pe) / (1 - pe) if (1 - pe) else 0.0

    return {
        'total': total, 'aciertos': aciertos, 'accuracy': accuracy,
        'precision': precision, 'recall': recall, 'f1': f1, 'kappa': kappa,
        **c,
    }


def evaluar(nombre, y_pred, y_true, clase_positiva='aislada', imprimir=True):
    '''
    Calcula y (opcionalmente) imprime confusión + métricas para el método.
    '''
    m = metricas_desde_confusion(confusion(y_pred, y_true))
    m['nombre'] = nombre

    if imprimir:
        print(f"\n── {nombre} " + "─" * max(2, 58 - len(nombre)))
        print(f"   Confusión (positivo = '{clase_positiva}'):")
        print(f"       TP={m['TP']:5d}   FP={m['FP']:5d}")
        print(f"       FN={m['FN']:5d}   TN={m['TN']:5d}")
        print(f"   Aciertos (TP+TN) : {m['aciertos']} / {m['total']}")
        print(f"   Accuracy         : {m['accuracy']:.4f}")
        print(f"   Precision        : {m['precision']:.4f}")
        print(f"   Recall           : {m['recall']:.4f}")
        print(f"   F1               : {m['f1']:.4f}")
        print(f"   Cohen's kappa    : {m['kappa']:.4f}")
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Barrido de parámetros — umbral del verde × umbral de distancia θ
# ──────────────────────────────────────────────────────────────────────────────

def barrer_verde(canal_rojo, canal_verde, positions, y_true,
                 umbrales_verde, umbrales_dist, umbral_rojo, usar_conectividad,
                 umbral_verde_bajo=None, verde_sigma=0.0, verde_cierre=0,
                 verde_rellenar=False):
    acum = {}
    shape = canal_verde.shape
    for uv in umbrales_verde:
        Mg, D_g, labels, labels_con_verde, t_g, labels_rojo, rojo_fuera = _verde_base(
            canal_rojo, canal_verde, uv, umbral_rojo, usar_conectividad,
            umbral_verde_bajo, verde_sigma, verde_cierre, verde_rellenar)
        clave_uv = 'otsu' if uv is None else round(float(uv), 4)
        for ud in umbrales_dist:
            mask, _, _, _, _ = _clasificar_verde_desde_campo(
                positions, D_g, labels, labels_con_verde, labels_rojo, rojo_fuera,
                ud, usar_conectividad, shape)
            acum[(clave_uv, ud)] = confusion(mask, y_true)
    return acum


def _fmt_param(v):
    '''Formatea un valor de parámetro (None/float/int/str) de forma compacta.'''
    if v is None:
        return 'otsu'
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def imprimir_tabla_barrido(acumulado, nombres_params, titulo, ruta_csv=None):
    '''
    Imprime la tabla del barrido.
    '''
    filas = []
    for params, c in acumulado.items():
        m = metricas_desde_confusion(c)
        filas.append((m['f1'], params, m))
    filas.sort(key=lambda r: r[0], reverse=True)
    best_f1 = filas[0][0]

    # ── Exportación a CSV (ordenada por F1, igual que la tabla impresa) ────────
    if ruta_csv:
        registros = []
        for f1, params, m in filas:
            fila = {nombre: p for nombre, p in zip(nombres_params, params)}
            fila.update({
                'accuracy':  round(m['accuracy'], 4),
                'precision': round(m['precision'], 4),
                'recall':    round(m['recall'], 4),
                'f1':        round(m['f1'], 4),
                'kappa':     round(m['kappa'], 4),
                'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN'], 'TN': m['TN'],
                'mejor': int(f1 == best_f1),
            })
            registros.append(fila)
        Path(ruta_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(registros).to_csv(ruta_csv, index=False, encoding='utf-8-sig')
        print(f"\n  [OK] Tabla de barrido exportada a: {ruta_csv}")

    ancho_par = "  ".join(f"{n:>8}" for n in nombres_params)
    print("\n" + "=" * 90)
    print(f"  {titulo}")
    print("=" * 90)
    print(f"  {ancho_par}   {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Kappa':>6}  "
          f"{'TP':>4} {'FP':>4} {'FN':>4}")
    print("  " + "-" * 86)

    for f1, params, m in filas:
        cols = "  ".join(f"{_fmt_param(p):>8}" for p in params)
        mark = "  <- MEJOR" if f1 == best_f1 else ""
        print(f"  {cols}   {m['accuracy']:>6.1%} {m['precision']:>6.1%} "
              f"{m['recall']:>6.1%} {m['f1']:>6.1%} {m['kappa']:>6.2f}  "
              f"{m['TP']:>4} {m['FP']:>4} {m['FN']:>4}{mark}")

    print("=" * 90)
    mejor_params, mejor_metricas = filas[0][1], filas[0][2]
    combo = "  ".join(f"{n}={_fmt_param(p)}" for n, p in zip(nombres_params, mejor_params))
    print(f"  Mejor F1={mejor_metricas['f1']:.1%}  ->  {combo}\n")
    return mejor_params, mejor_metricas


# ──────────────────────────────────────────────────────────────────────────────
# Reporte de falsos positivos / negativos (modo par con etiquetas)
# ──────────────────────────────────────────────────────────────────────────────

def reporte_fallos(nombre, positions, clases_raw, y_pred, y_true, extra_cols=None):
    '''
    Lista las partículas mal clasificadas (FP y FN) con sus coordenadas.
    '''
    y_pred = np.asarray(y_pred, dtype=bool)
    y_true = np.asarray(y_true, dtype=bool)
    idx_fp = np.where( y_pred & ~y_true)[0]
    idx_fn = np.where(~y_pred &  y_true)[0]
    extra_cols = extra_cols or {}

    def _bloque(idxs, titulo):
        if len(idxs) == 0:
            return
        print(f"\n  [{nombre}] {titulo} ({len(idxs)}):")
        cab = f"  {'#':>4}  {'y':>6}  {'x':>6}"
        for k in extra_cols:
            cab += f"  {k:>10}"
        cab += "  clase_real"
        print(cab)
        print("  " + "-" * (len(cab)))
        for j, idx in enumerate(idxs):
            y, x = positions[idx]
            linea = f"  {j+1:>4}  {y:>6}  {x:>6}"
            for arr in extra_cols.values():
                linea += f"  {arr[idx]:>10}"
            linea += f"  {clases_raw[idx]}"
            print(linea)

    _bloque(idx_fp, "FALSOS POSITIVOS (predichas aisladas, no lo son)")
    _bloque(idx_fn, "FALSOS NEGATIVOS (aisladas no detectadas)")


# ──────────────────────────────────────────────────────────────────────────────
# Figura de diagnóstico (opcional)
# ──────────────────────────────────────────────────────────────────────────────

def guardar_figura_diagnostico(canal_rojo, canal_verde, positions, y_true, mask,
                               args, ruta_salida):
    '''
    Dibuja el overlay R+G, el borde de la traza verde y la banda de θ.
    '''
    Mg, t_g = construir_mascara_verde(
        canal_verde, args.umbral_verde, args.umbral_verde_bajo,
        args.verde_sigma, args.verde_cierre, args.verde_rellenar)
    umbral_dist = args.umbral_dist
    D_g = ndi.distance_transform_edt(~Mg)

    pos = np.array(positions)
    ys, xs = pos[:, 0], pos[:, 1]

    H, W = canal_rojo.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0] = canal_rojo
    rgb[..., 1] = canal_verde

    fig, ax = plt.subplots(1, 1, figsize=(9, 9), constrained_layout=True)
    ax.imshow(rgb, vmin=0, vmax=1)
    ax.contour(Mg.astype(float), levels=[0.5], colors='#2ecc71', linewidths=0.6)
    if umbral_dist > 0:
        ax.contour(D_g, levels=[float(umbral_dist)], colors='#f1c40f',
                   linewidths=0.8, linestyles='--')

    tp =  mask &  y_true
    fp =  mask & ~y_true
    fn = ~mask &  y_true
    tn = ~mask & ~y_true
    ax.scatter(xs[tn], ys[tn], s=6,  c='#7f8c8d', label='TN', alpha=0.5)
    ax.scatter(xs[tp], ys[tp], s=16, c='#2ecc71', label='TP', edgecolors='k', linewidths=0.3)
    ax.scatter(xs[fp], ys[fp], s=16, c='#f1c40f', label='FP', edgecolors='k', linewidths=0.3)
    ax.scatter(xs[fn], ys[fn], s=16, c='#e74c3c', label='FN', edgecolors='k', linewidths=0.3)
    ax.set_title("EDT al verde — verde=TP, amarillo=FP, rojo=FN, gris=TN\n"
                 "(línea verde = traza; discontinua amarilla = banda θ)",
                 fontsize=10, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=7, framealpha=0.7)

    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] Figura de diagnóstico guardada en: {ruta_salida}")


# ──────────────────────────────────────────────────────────────────────────────
# Pintar el campo de la EDT (para ver cómo funciona)
# ──────────────────────────────────────────────────────────────────────────────

def guardar_campo_edt(tif_path, args, ruta_salida, positions=None, imprimir=True):
    '''
    Pinta el campo de distancia de la EDT.
    '''
    canal_rojo, canal_verde = load_tif_image(tif_path)

    if args.edt_sobre == 'rojo':
        t = _umbral_auto(canal_rojo, args.umbral_rojo)
        M = canal_rojo > t
        D = ndi.distance_transform_edt(M)
        titulo = "EDT del rojo — distancia de cada píxel de partícula al fondo (px)"
    else:
        Mg, t = construir_mascara_verde(
            canal_verde, args.umbral_verde, args.umbral_verde_bajo,
            args.verde_sigma, args.verde_cierre, args.verde_rellenar)
        D = ndi.distance_transform_edt(~Mg)
        titulo = "EDT al verde — distancia de cada píxel a la traza (px)"

    fig, ax = plt.subplots(figsize=(9.5, 9), constrained_layout=True)

    if args.edt_cortes:
        cortes = sorted(float(c) for c in args.edt_cortes)
        n_cat  = len(cortes) + 2
        if args.edt_colores and len(args.edt_colores) >= n_cat:
            paleta = np.array([to_rgb(c) for c in args.edt_colores[:n_cat]])
        else:
            base = args.edt_colores if args.edt_colores else ['#2ecc71', '#e67e22', '#e74c3c']
            cmap = LinearSegmentedColormap.from_list('vnr', base)
            paleta = cmap(np.linspace(0, 1, n_cat))[:, :3]

        idx = np.zeros(D.shape, dtype=int)
        pos_mask = D > 0
        idx[pos_mask] = 1 + np.digitize(D[pos_mask], cortes)
        rgb = paleta[np.clip(idx, 0, n_cat - 1)]
        ax.imshow(rgb)

        etiquetas = ["d = 0"]
        for k in range(len(cortes)):
            ini = "0" if k == 0 else f"{cortes[k-1]:g}"
            etiquetas.append(f"{ini} < d ≤ {cortes[k]:g}")
        etiquetas.append(f"d > {cortes[-1]:g}")
        parches = [mpatches.Patch(color=paleta[i], label=etiquetas[i]) for i in range(n_cat)]
        ax.legend(handles=parches, loc='upper right', fontsize=8, framealpha=0.8)
    else:
        cmap = LinearSegmentedColormap.from_list(
            'verde_naranja_rojo', ['#2ecc71', '#e67e22', '#e74c3c'])
        im = ax.imshow(D, cmap=cmap)
        cb = fig.colorbar(im, ax=ax, shrink=0.82)
        cb.set_label('distancia (px)')

    if positions is not None and len(positions) > 0:
        pos = np.array(positions)
        ax.scatter(pos[:, 1], pos[:, 0], s=10, c='white',
                   edgecolors='k', linewidths=0.4, label='partículas')

    ax.set_title(titulo, fontsize=10, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=130, bbox_inches='tight')
    plt.close(fig)
    if imprimir:
        modo = "discreto" if args.edt_cortes else "continuo"
        print(f"\n[OK] Campo EDT ({args.edt_sobre}, modo {modo}) guardado en: {ruta_salida}")


# ──────────────────────────────────────────────────────────────────────────────
# Diagnóstico de máscaras (para depurar: ¿es Mg de verdad la traza?)
# ──────────────────────────────────────────────────────────────────────────────

def guardar_diagnostico_mascaras(tif_path, args, ruta_salida, positions=None,
                                 imprimir=True):
    '''
    Pinta y resume las máscaras binarias del verde (Mg) y del rojo (Mr).
    '''
    canal_rojo, canal_verde = load_tif_image(tif_path)
    H, W = canal_verde.shape

    Mg, t_g = construir_mascara_verde(
        canal_verde, args.umbral_verde, args.umbral_verde_bajo,
        args.verde_sigma, args.verde_cierre, args.verde_rellenar)
    t_r = _umbral_auto(canal_rojo, args.umbral_rojo)
    Mr = canal_rojo > t_r
    cob_g = 100.0 * Mg.mean()
    cob_r = 100.0 * Mr.mean()

    D_g = ndi.distance_transform_edt(~Mg)

    if imprimir:
        print(f"\n{'─'*70}")
        print(f"  Diagnóstico de máscaras — {os.path.basename(tif_path)}")
        print(f"  Imagen : {W} x {H} px")
        print(f"  Verde  : umbral t_g={t_g:.4f}  ->  Mg cubre {cob_g:.1f}% de la imagen")
        print(f"  Rojo   : umbral t_r={t_r:.4f}  ->  Mr cubre {cob_r:.1f}% de la imagen")
        if positions is not None and len(positions) > 0:
            ds = np.array([D_g[min(max(int(y), 0), H - 1),
                               min(max(int(x), 0), W - 1)] for (y, x) in positions])
            th = args.umbral_dist
            print(f"  Distancia al verde en las partículas (px): "
                  f"min={ds.min():.1f}  mediana={np.median(ds):.1f}  "
                  f"p90={np.percentile(ds, 90):.1f}  max={ds.max():.1f}")
            print(f"  Con θ={th:g}: {100.0*np.mean(ds <= th):.1f}% de partículas "
                  f"quedarían DENTRO (no aisladas).")

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0] = canal_rojo
    rgb[..., 1] = canal_verde

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    axes[0].imshow(rgb, vmin=0, vmax=1)
    axes[0].imshow(np.ma.masked_where(~Mg, Mg.astype(float)), cmap='cool', alpha=0.35)
    if positions is not None and len(positions) > 0:
        pos = np.array(positions)
        axes[0].scatter(pos[:, 1], pos[:, 0], s=8, c='white',
                        edgecolors='k', linewidths=0.3)
    axes[0].set_title(f"Overlay R+G + máscara verde Mg (cian)  [{cob_g:.1f}%]",
                      fontsize=10, fontweight='bold')
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(canal_verde, cmap='gray', vmin=0, vmax=1)
    axes[1].contour(Mg.astype(float), levels=[0.5], colors='#2ecc71', linewidths=0.6)
    axes[1].set_title(f"Canal verde crudo + contorno de Mg (t_g={t_g:.3f})",
                      fontsize=10, fontweight='bold')
    axes[1].set_xticks([]); axes[1].set_yticks([])

    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=130, bbox_inches='tight')
    plt.close(fig)
    if imprimir:
        print(f"  [OK] Diagnóstico de máscaras guardado en: {ruta_salida}")


# ──────────────────────────────────────────────────────────────────────────────
# Procesado de un par (tif, csv): evaluación con parámetros fijos
# ──────────────────────────────────────────────────────────────────────────────

def procesar_par(tif_path, csv_path, args, out_dir=None, imprimir=True):
    '''
    Carga, clasifica y evalúa de forma jerárquica (Etapa 1: Aisladas, Etapa 2: Int vs Borde).
    Returns: dict con métricas de ambas etapas.
    '''
    if imprimir:
        print(f"\n{'─'*70}")
        print(f"  Imagen : {tif_path}")
        print(f"  CSV    : {csv_path}")

    canal_rojo, canal_verde = load_tif_image(tif_path)
    positions, clases_raw, y_true = cargar_ground_truth(csv_path)
    tiene_etiquetas = y_true is not None

    mask, info = clasificar_aisladas_verde(
        canal_rojo, canal_verde, positions,
        umbral_verde=args.umbral_verde, umbral_rojo=args.umbral_rojo,
        umbral_dist=args.umbral_dist, usar_conectividad=args.conectividad,
        umbral_verde_bajo=args.umbral_verde_bajo, verde_sigma=args.verde_sigma,
        verde_cierre=args.verde_cierre, verde_rellenar=args.verde_rellenar)

    if imprimir:
        modo = "evaluación jerárquica" if tiene_etiquetas else "inferencia (sin clase)"
        n_aisl = int(np.sum(y_true)) if tiene_etiquetas else 0
        print(f"  Tamaño : {canal_verde.shape[1]} x {canal_verde.shape[0]} px  |  "
              f"Partículas: {len(positions)}  |  Modo: {modo}")
        print(f"  Verde: umbral={info['umbral_verde']:.4f}  θ={info['umbral_dist']:g}  "
              f"en_verde={int(np.sum(info['en_verde']))}")
        if tiene_etiquetas:
            print(f"  Aisladas (verdad): {n_aisl} "
                  f"({100.0 * n_aisl / max(1, len(positions)):.1f}%)")

    conf = None
    if tiene_etiquetas:
        y_true_interior = np.array([c in CLASES_INTERIOR for c in clases_raw], dtype=bool)
        y_true_borde    = np.array([c in CLASES_BORDE for c in clases_raw], dtype=bool)

        if imprimir:
            print(f"\n{'='*50}")
            print("  EVALUACIÓN JERÁRQUICA (2 ETAPAS)")
            print(f"{'='*50}")
        
        # --- ETAPA 1: Aisladas vs Resto ---
        res_aislada = evaluar("Etapa 1: AISLADA vs RESTO", mask, y_true, clase_positiva='aislada', imprimir=imprimir)
        
        # --- ETAPA 2: Interior vs Borde ---
        # Filtramos para quedarnos estrictamente con la población que NO es verdaderamente aislada
        filtro_no_aisladas = y_true_interior | y_true_borde
        
        if np.any(filtro_no_aisladas):
            y_pred_interior_fil = info['interior'][filtro_no_aisladas]
            y_true_interior_fil = y_true_interior[filtro_no_aisladas]
            
            res_int_borde = evaluar("Etapa 2: INTERIOR vs BORDE", 
                                    y_pred_interior_fil, y_true_interior_fil, 
                                    clase_positiva='interior', imprimir=imprimir)
        else:
            res_int_borde = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            if imprimir:
                print("\n── Etapa 2: INTERIOR vs BORDE ──────────────────────")
                print("   [AVISO] El CSV no tiene partículas de interior o borde para evaluar.")
        
        # Agrupar datos para devolver al bucle principal
        conf = {
            'aislada':           {k: res_aislada[k]   for k in ('TP', 'TN', 'FP', 'FN')},
            'interior_vs_borde': {k: res_int_borde[k] for k in ('TP', 'TN', 'FP', 'FN')}
        }
        
        if imprimir and args.mostrar_fallos:
            reporte_fallos("Etapa 1", positions, clases_raw, mask, y_true,
                           extra_cols={'d_verde': np.round(info['dist_verde'], 1)})
    elif imprimir:
        print(f"  Aisladas detectadas: {int(mask.sum())}/{len(positions)}")

    # ── Guardado del detalle por partícula ────────────────────────────────────
    if args.guardar_csv or out_dir:
        df_out = pd.DataFrame({
            'y': [p[0] for p in positions],
            'x': [p[1] for p in positions],
        })
        if tiene_etiquetas:
            df_out['clase']       = clases_raw
            df_out['gt_aislada']  = y_true.astype(int)
            df_out['gt_interior'] = y_true_interior.astype(int)  
            df_out['gt_borde']    = y_true_borde.astype(int)     
            
        df_out['V_aislada']    = mask.astype(int)
        df_out['V_interior']   = info['interior'].astype(int)
        df_out['V_borde']      = info['borde'].astype(int)
        df_out['V_dist_verde'] = np.round(info['dist_verde'], 2)
        df_out['V_en_verde']   = info['en_verde'].astype(int)

        if out_dir:
            base = os.path.splitext(os.path.basename(csv_path))[0]
            ruta = os.path.join(out_dir, base + "_deteccion.csv")
        else:
            ruta = args.guardar_csv
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(ruta, index=False, encoding='utf-8-sig')
        if imprimir:
            print(f"  Guardado: {ruta}")

    # ── Figura (solo par individual con etiquetas) ────────────────────────────
    if args.guardar_figura and tiene_etiquetas and out_dir is None:
        guardar_figura_diagnostico(canal_rojo, canal_verde, positions, y_true,
                                   mask, args, args.guardar_figura)

    return conf


# ──────────────────────────────────────────────────────────────────────────────
# Filtrado -> CSV con columnas X, Y, clase, Filtradas
# ──────────────────────────────────────────────────────────────────────────────

ETIQUETA_FILTRADA = 'aislada'

def filtrar_particulas(tif_path, csv_path, args, ruta_salida=None, imprimir=True):
    '''
    Genera un CSV filtrando partículas y sumando las nuevas columnas.
    '''
    canal_rojo, canal_verde = load_tif_image(tif_path)
    positions, _, _ = cargar_ground_truth(csv_path)

    mask, info = clasificar_aisladas_verde(
        canal_rojo, canal_verde, positions,
        umbral_verde=args.umbral_verde, umbral_rojo=args.umbral_rojo,
        umbral_dist=args.umbral_dist, usar_conectividad=args.conectividad,
        umbral_verde_bajo=args.umbral_verde_bajo, verde_sigma=args.verde_sigma,
        verde_cierre=args.verde_cierre, verde_rellenar=args.verde_rellenar)

    filtradas = np.where(mask, ETIQUETA_FILTRADA, '')

    df_orig = pd.read_csv(csv_path, encoding='utf-8-sig')
    mapa = {c.strip().lower(): c for c in df_orig.columns}
    if 'x' not in mapa or 'y' not in mapa:
        raise ValueError("El CSV debe contener columnas X e Y.")
    if len(df_orig) != len(positions):
        raise ValueError("Desajuste de filas entre el CSV y las posiciones cargadas.")

    col_x, col_y = mapa['x'], mapa['y']
    col_clase    = mapa.get('clase')

    df_out = pd.DataFrame()
    df_out['X']         = df_orig[col_x].to_numpy()
    df_out['Y']         = df_orig[col_y].to_numpy()
    df_out['clase']     = df_orig[col_clase].to_numpy() if col_clase else ''
    df_out['Filtradas'] = filtradas
    
    # Nuevas columnas de clasificación añadidas al filtro
    df_out['V_interior'] = info['interior'].astype(int)
    df_out['V_borde']    = info['borde'].astype(int)

    if ruta_salida is None:
        base = os.path.splitext(csv_path)[0]
        ruta_salida = base + "_filtrado.csv"
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(ruta_salida, index=False, encoding='utf-8-sig')

    if imprimir:
        n_aisl = int(mask.sum())
        n_tot  = len(positions)
        print(f"\n{'─'*70}")
        print(f"  Filtrado (EDT al verde)")
        print(f"  Imagen : {tif_path}")
        print(f"  CSV    : {csv_path}")
        print(f"  Aisladas marcadas: {n_aisl}/{n_tot} "
              f"({100.0 * n_aisl / max(1, n_tot):.1f}%)")
        if col_clase is None:
            print("  [AVISO] El CSV no tenía columna 'clase'; se escribe vacía.")
        print(f"  Guardado : {ruta_salida}")

    return ruta_salida


# ──────────────────────────────────────────────────────────────────────────────
# Procesado de un par: barrido del método
# ──────────────────────────────────────────────────────────────────────────────

def barrer_par(tif_path, csv_path, args):
    canal_rojo, canal_verde = load_tif_image(tif_path)
    positions, clases_raw, y_true = cargar_ground_truth(csv_path)
    if y_true is None:
        print(f"  [AVISO] {os.path.basename(csv_path)} no tiene columna 'clase'; "
              "se omite del barrido.")
        return None

    umbrales_verde = args.umbrales_verde if args.umbrales_verde is not None else REJILLA_UMBRAL_VERDE
    umbrales_dist  = args.umbrales_dist  if args.umbrales_dist  is not None else REJILLA_UMBRAL_DIST

    return barrer_verde(canal_rojo, canal_verde, positions, y_true,
                        umbrales_verde, umbrales_dist,
                        args.umbral_rojo, args.conectividad,
                        args.umbral_verde_bajo, args.verde_sigma,
                        args.verde_cierre, args.verde_rellenar)


def sumar_acumulados(dst, src):
    for clave, c in src.items():
        if clave not in dst:
            dst[clave] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for k in ('TP', 'TN', 'FP', 'FN'):
            dst[clave][k] += c[k]


# ──────────────────────────────────────────────────────────────────────────────
# Búsqueda de pares en un directorio
# ──────────────────────────────────────────────────────────────────────────────

def buscar_pares(directorio):
    csv_files = sorted(glob.glob(os.path.join(directorio, "*.csv")))
    tif_paths, csv_paths, omitidos = [], [], []

    for csv_path in csv_files:
        base = os.path.splitext(csv_path)[0]
        if base.endswith("_deteccion") or base.endswith("_filtrado"):
            continue
        tif_path = base + ".tif"
        if os.path.isfile(tif_path):
            tif_paths.append(tif_path)
            csv_paths.append(csv_path)
        else:
            omitidos.append(csv_path)

    if omitidos:
        print("\n  [AVISO] CSV sin .tif del mismo nombre (se omiten):")
        for p in omitidos:
            print(f"    {os.path.basename(p)}")
    return tif_paths, csv_paths


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def construir_parser():
    parser = argparse.ArgumentParser(
        description="Detecta partículas aisladas con la EDT aplicada al canal "
                    "verde (distancia a las trazas), con barrido de parámetros, "
                    "modo directorio, inferencia y filtrado.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Fuente de datos ───────────────────────────────────────────────────────
    parser.add_argument("--dir", default=None,
                        help="Procesa todos los pares .tif + .csv del directorio.")
    parser.add_argument("--tif", default=None, help="Ruta al .tif (modo archivo).")
    parser.add_argument("--csv", default=None, help="Ruta al CSV (modo archivo).")

    # ── Parámetros fijos del método ───────────────────────────────────────────
    parser.add_argument("--umbral-verde", type=float, default=None,
                        help="Umbral de binarización del verde. None -> Otsu.")
    parser.add_argument("--umbral-rojo", type=float, default=None,
                        help="Umbral del rojo (solo conectividad). None -> Otsu.")
    parser.add_argument("--umbral-dist", type=float, default=2.0,
                        help="Umbral de distancia θ (px). aislada ⟺ d_g > θ.")
    parser.add_argument("--sin-conectividad", dest="conectividad",
                        action="store_false",
                        help="No comprobar si la isla roja conecta con el verde; "
                             "decidir solo por la distancia al verde.")
    parser.set_defaults(conectividad=True)

    # ── Segmentación del verde (cobertura de Mg) ──────────────────────────────
    parser.add_argument("--umbral-verde-bajo", type=float, default=None,
                        help="Umbral BAJO de histéresis del verde. Si se indica, Mg "
                             "crece desde los núcleos (umbral alto) hasta este umbral "
                             "bajo por conectividad, capturando los halos tenues.")
    parser.add_argument("--verde-sigma", type=float, default=0.0,
                        help="Suavizado gaussiano del verde antes de umbralar (0 = off).")
    parser.add_argument("--verde-cierre", type=int, default=0,
                        help="Iteraciones de cierre morfológico de Mg (rellena huecos "
                             "pequeños y conecta). 0 = off.")
    parser.add_argument("--verde-rellenar", action="store_true",
                        help="Rellenar los huecos interiores de Mg.")

    # ── Barrido ───────────────────────────────────────────────────────────────
    parser.add_argument("--barrer", action="store_true",
                        help="Barre umbral del verde × umbral de distancia θ "
                             "(requiere columna 'clase').")
    parser.add_argument("--umbrales-verde", type=float, nargs="+", default=None,
                        metavar="U", help="Rejilla de umbral del verde.")
    parser.add_argument("--umbrales-dist", type=float, nargs="+", default=None,
                        metavar="D", help="Rejilla de umbral de distancia θ (px).")

    # ── Filtrado ──────────────────────────────────────────────────────────────
    parser.add_argument("--filtrar", action="store_true",
                        help="Genera <base>_filtrado.csv con columnas "
                             "X, Y, clase, Filtradas.")
    parser.add_argument("--guardar-filtrado", default=None,
                        help="Ruta explícita del CSV filtrado (modo archivo). "
                             "Por defecto: <base>_filtrado.csv junto al CSV.")

    # ── Salidas ───────────────────────────────────────────────────────────────
    parser.add_argument("--guardar-csv", default=None,
                        help="Ruta del CSV de detalle por partícula (modo archivo).")
    parser.add_argument("--guardar-figura", default=None,
                        help="Ruta de la figura de diagnóstico (modo archivo).")
    parser.add_argument("--mostrar-fallos", action="store_true",
                        help="Lista FP y FN con coordenadas (modo archivo con etiquetas).")
    parser.add_argument("--guardar-barrido", default=None,
                        help="Exporta la tabla de barrido a este CSV. Solo con --barrer.")

    # ── Pintar el campo de la EDT ─────────────────────────────────────────────
    parser.add_argument("--guardar-edt", default=None,
                        help="Guarda una imagen del campo EDT en esta ruta (necesita --tif).")
    parser.add_argument("--edt-sobre", choices=["verde", "rojo"], default="verde",
                        help="Sobre qué máscara pintar la EDT: 'verde' (distancia a "
                             "la traza) o 'rojo' (distancia de los blobs al fondo).")
    parser.add_argument("--edt-cortes", type=float, nargs="+", default=None,
                        metavar="C", help="Cortes en px para el modo discreto. Si se "
                             "omite, mapa continuo con barra de color.")
    parser.add_argument("--edt-colores", nargs="+", default=None, metavar="HEX",
                        help="Colores hex del modo discreto (uno para d=0 y uno por tramo).")

    # ── Diagnóstico de máscaras ───────────────────────────────────────────────
    parser.add_argument("--guardar-mascaras", default=None,
                        help="Guarda un diagnóstico de las máscaras Mg/Mr (necesita --tif).")
    return parser


def main(argv=None):
    args = construir_parser().parse_args(argv)

    # ══════════════════════════════════════════════════════════════════════════
    # PINTAR EL CAMPO DE LA EDT
    # ══════════════════════════════════════════════════════════════════════════
    if args.guardar_edt:
        if not args.tif or not os.path.isfile(args.tif):
            print("[ERROR] Para pintar la EDT indica un --tif válido.")
            sys.exit(1)
        positions = None
        if args.csv and os.path.isfile(args.csv):
            try:
                positions, _, _ = cargar_ground_truth(args.csv)
            except Exception:
                positions = None
        guardar_campo_edt(args.tif, args, args.guardar_edt, positions=positions)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNÓSTICO DE MÁSCARAS
    # ══════════════════════════════════════════════════════════════════════════
    if args.guardar_mascaras:
        if not args.tif or not os.path.isfile(args.tif):
            print("[ERROR] Para el diagnóstico de máscaras indica un --tif válido.")
            sys.exit(1)
        positions = None
        if args.csv and os.path.isfile(args.csv):
            try:
                positions, _, _ = cargar_ground_truth(args.csv)
            except Exception:
                positions = None
        guardar_diagnostico_mascaras(args.tif, args, args.guardar_mascaras,
                                     positions=positions)
        return

    if not args.dir and not (args.tif and args.csv):
        print("[ERROR] Indica --dir DIRECTORIO, o bien --tif y --csv.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════════
    # MODO DIRECTORIO
    # ══════════════════════════════════════════════════════════════════════════
    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"[ERROR] No es un directorio válido: {args.dir}")
            sys.exit(1)

        tif_paths, csv_paths = buscar_pares(args.dir)
        if not tif_paths:
            print("[ERROR] No se encontró ningún par .csv + .tif.")
            sys.exit(1)

        print(f"\n  Directorio        : {os.path.abspath(args.dir)}")
        print(f"  Pares encontrados : {len(tif_paths)}")

        # ── Filtrado (un <base>_filtrado.csv por par) ─────────────────────────
        if args.filtrar:
            print(f"\n  Filtrando (EDT al verde) sobre {len(tif_paths)} par/es...")
            for tif_path, csv_path in zip(tif_paths, csv_paths):
                try:
                    filtrar_particulas(tif_path, csv_path, args)
                except Exception as e:
                    print(f"  [ERROR] {os.path.basename(csv_path)}: {e}")
            return

        # ── Barrido GLOBAL acumulado ──────────────────────────────────────────
        if args.barrer:
            print("\n  Acumulando barrido sobre todas las imágenes con etiquetas...\n")
            glob_acum = {}
            usados = 0
            for tif_path, csv_path in zip(tif_paths, csv_paths):
                print(f"  Barriendo: {os.path.basename(csv_path)}")
                try:
                    ac = barrer_par(tif_path, csv_path, args)
                    if ac is None:
                        continue
                    sumar_acumulados(glob_acum, ac)
                    usados += 1
                except Exception as e:
                    print(f"  [ERROR] {os.path.basename(csv_path)}: {e}")

            if usados == 0:
                print("\n  [ERROR] Ningún archivo tenía columna 'clase'.")
                sys.exit(1)

            print(f"\n  Barrido global sobre {usados} imagen/es.")
            imprimir_tabla_barrido(
                glob_acum, ["umbVerde", "umbDist"],
                f"BARRIDO GLOBAL — EDT al verde ({usados} imágenes)",
                ruta_csv=args.guardar_barrido)
            return

        # ── Evaluación con parámetros fijos por cada par ──────────────────────
        glob_acum = {
            'aislada':           {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
            'interior_vs_borde': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        }
        con_etiquetas = 0
        for tif_path, csv_path in zip(tif_paths, csv_paths):
            try:
                conf = procesar_par(tif_path, csv_path, args, out_dir=args.dir)
                if conf is not None:
                    con_etiquetas += 1
                    for clase in ('aislada', 'interior_vs_borde'):
                        for k in ('TP', 'TN', 'FP', 'FN'):
                            glob_acum[clase][k] += conf[clase][k]
            except Exception as e:
                print(f"  [ERROR] {os.path.basename(csv_path)}: {e}")

        print(f"\n{'═'*70}")
        print(f"  RESUMEN GLOBAL — {len(tif_paths)} par/es ({con_etiquetas} con etiquetas)")
        print(f"{'═'*70}")
        if con_etiquetas > 0:
            m1 = metricas_desde_confusion(glob_acum['aislada'])
            print(f"  Etapa 1 (Aisladas)   : aciertos {m1['aciertos']}/{m1['total']}  "
                  f"Acc {m1['accuracy']:.1%}  Prec {m1['precision']:.1%}  "
                  f"Rec {m1['recall']:.1%}  F1 {m1['f1']:.1%}  kappa {m1['kappa']:.2f}")
            
            m2 = metricas_desde_confusion(glob_acum['interior_vs_borde'])
            if m2['total'] > 0:
                print(f"  Etapa 2 (Int vs Bor): aciertos {m2['aciertos']}/{m2['total']}  "
                      f"Acc {m2['accuracy']:.1%}  Prec {m2['precision']:.1%}  "
                      f"Rec {m2['recall']:.1%}  F1 {m2['f1']:.1%}  kappa {m2['kappa']:.2f}")
        else:
            print("  Ningún archivo tenía columna 'clase' — sin métricas globales.")
        print(f"{'═'*70}\n")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MODO ARCHIVO INDIVIDUAL
    # ══════════════════════════════════════════════════════════════════════════
    for path in (args.tif, args.csv):
        if not os.path.isfile(path):
            print(f"[ERROR] No se encuentra: {path}")
            sys.exit(1)

    # ── Filtrado -> <base>_filtrado.csv ───────────────────────────────────────
    if args.filtrar:
        filtrar_particulas(args.tif, args.csv, args, ruta_salida=args.guardar_filtrado)
        return

    # ── Barrido sobre par individual ──────────────────────────────────────────
    if args.barrer:
        print(f"\n  Barrido sobre par individual:\n   {args.tif}\n   {args.csv}")
        ac = barrer_par(args.tif, args.csv, args)
        if ac is None:
            print("[ERROR] El CSV no tiene columna 'clase'; el barrido la requiere.")
            sys.exit(1)
        imprimir_tabla_barrido(
            ac, ["umbVerde", "umbDist"], "BARRIDO — EDT al verde",
            ruta_csv=args.guardar_barrido)
        return

    # ── Evaluación / inferencia con parámetros fijos ──────────────────────────
    procesar_par(args.tif, args.csv, args, out_dir=None)


if __name__ == "__main__":
    main()