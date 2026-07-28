"""
clasificacion_EDT.py 
================================================================================
Clasificador clasico en DOS ETAPAS de trayectorias en imagenes de dos canales
(rojo = trayectorias estaticas; verde = trazas registradas)

--------------------------------------------------------------------------------
METODO
--------------------------------------------------------------------------------
Se construye, por trayectoria i, su TERRITORIO rojo R_i mediante watershed del
canal rojo sembrado con los centroides anotados (un territorio por trayectoria,
sin blobs compartidos). Sobre la mascara verde M_g = {I_g > t_g} y
su transformada de distancia se definen dos estadisticos:

    ETAPA 1 (aislada / no aislada)  —  criterio topologico, sin parametros libres
        d_min(i) = min_{p in R_i} D_g^out(p)      D_g^out = EDT( ~M_g )
        aislada  <=>  d_min(i) > theta            (theta = 0)

    ETAPA 2 (interior / borde)  —  solo sobre las NO aisladas
        prof(i)  = D_g^in(c_i)                     D_g^in = EDT( M_g )
        interior <=>  prof(i) >= p_prof            (p_prof = 4, t_g = 12)

La misma transformada de distancia euclidea al verde resuelve ambas etapas: la
etapa 1 la lee HACIA FUERA del verde (distancia del blob a la traza), la etapa 2
HACIA DENTRO (profundidad del centroide en la traza).

--------------------------------------------------------------------------------
USO
--------------------------------------------------------------------------------
    python clasificacion_EDT.py --tif img.tif --csv datos.csv
    python clasificacion_EDT.py --dir /ruta/ --evaluar-completo
    python clasificacion_EDT.py --dir /ruta/ --filtrar
    python clasificacion_EDT.py --tif img.tif --csv datos.csv --barrer
================================================================================
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import os
import sys
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from scipy import ndimage as ndi

from funciones_auxiliares import construir_territorios

import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Etiquetas de la verdad de campo
# ──────────────────────────────────────────────────────────────────────────────

CLASES_AISLADAS = {'aislada', 'aislado'}
CLASES_INTERIOR = {'interior'}
CLASES_BORDE    = {'borde'}
ETIQUETA_FILTRADA = 'aislada'


# ──────────────────────────────────────────────────────────────────────────────
# Parametros por defecto del metodo (fijados empiricamente)
# ──────────────────────────────────────────────────────────────────────────────

UMBRAL_DIST_DEF   = 0.0     # theta  : etapa 1, aislada <=> d_min > theta
UMBRAL_VERDE_DEF  = 0.0     # t_g etapa 1 : M_g = {I_g > 0} (soporte del verde)
FACTOR_ROJO_DEF   = 0.25    # t_r = factor * Otsu(rojo>0) para el watershed
TG_INTERIOR_DEF   = 12.0    # t_g etapa 2 : mascara verde para la profundidad
PROF_INTERIOR_DEF = 4.0     # p_prof : interior <=> prof >= p_prof

# Rejillas de barrido (etapa 1)
REJILLA_UMBRAL_DIST = [0, 1, 2, 3, 5]


# ──────────────────────────────────────────────────────────────────────────────
# Carga de imagen CRUDA (sin normalizar)
# ──────────────────────────────────────────────────────────────────────────────

def load_tif_crudo(tif_path):
    '''
    Carga el TIF de 2 canales SIN normalizar. Necesario porque el criterio usa
    umbrales en cuentas (t_g, Otsu del rojo) y el maximo de verde por territorio.

    Returns:
        (canal_rojo, canal_verde) : dos np.ndarray (H, W) con valores crudos.
    '''
    image = tifffile.imread(tif_path)
    if image.ndim == 3 and image.shape[2] == 2:
        image = np.moveaxis(image, -1, 0)
    if image.shape[0] != 2:
        raise ValueError(f"Se esperaban 2 canales; forma {image.shape}")
    return image[0].astype(np.float32), image[1].astype(np.float32)



def cargar_ground_truth(csv_path):
    '''
    Lee el CSV. Soporta con clase (x, y, clase) o sin clase (x, y).

    Returns:
        (positions, clases_raw, y_true)
        - positions  : lista de (y, x) enteros (fila, columna).
        - clases_raw : lista de str, o None si no hay columna clase.
        - y_true     : np.ndarray bool (True=aislada), o None.
    '''
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = [c.strip().lower() for c in df.columns]
    if not {'x', 'y'}.issubset(df.columns):
        raise ValueError("El CSV debe contener al menos las columnas: x, y")

    yy = pd.to_numeric(df['y'], errors='coerce').round()
    xx = pd.to_numeric(df['x'], errors='coerce').round()
    positions = list(zip(yy.astype('Int64'), xx.astype('Int64')))
    positions = [(int(y), int(x)) for y, x in positions]

    if 'clase' not in df.columns:
        return positions, None, None

    df['clase'] = df['clase'].astype(str).str.strip().str.lower()
    clases_raw = df['clase'].tolist()
    y_true = np.array([c in CLASES_AISLADAS for c in clases_raw], dtype=bool)
    return positions, clases_raw, y_true


# ──────────────────────────────────────────────────────────────────────────────
# Estadisticos y clasificacion en 2 etapas
# ──────────────────────────────────────────────────────────────────────────────

def clasificar_edt(canal_rojo, canal_verde, positions,
                   umbral_dist=UMBRAL_DIST_DEF, umbral_verde=UMBRAL_VERDE_DEF,
                   factor_rojo=FACTOR_ROJO_DEF,
                   tg_interior=TG_INTERIOR_DEF, prof_interior=PROF_INTERIOR_DEF):
    '''
    Clasificador completo en 2 etapas. Los canales deben venir CRUDOS.

    Returns:
        (mask_aisladas, info) 
        info incluye: 'interior', 'borde' (mascaras bool sobre TODAS las
        trayectorias; interior/borde solo True en no-aisladas), 'dist_verde'
        (= d_min, alias retro-compatible), 'en_verde' (= ~aislada), 'prof',
        'g_max', 'area', 'umbral_verde' (t_g etapa 1), 'umbral_dist'.
    '''
    n = len(positions)
    H, W = canal_verde.shape
    filas = np.clip(np.array([int(p[0]) for p in positions]), 0, H - 1)
    cols  = np.clip(np.array([int(p[1]) for p in positions]), 0, W - 1)

    # ── ETAPA 1 : mascara de soporte del verde y d_min por territorio ─────────
    Mg1 = canal_verde > umbral_verde
    D_out = ndi.distance_transform_edt(~Mg1) if Mg1.any() \
        else np.full(canal_verde.shape, np.inf)

    lab, t_r = construir_territorios(canal_rojo, filas, cols, factor_rojo)
    idx = np.arange(1, n + 1)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(int)

    if np.isinf(D_out).all():
        d_min = np.full(n, np.inf)
    else:
        d_min = np.asarray(ndi.minimum(D_out, lab, index=idx), dtype=float)
    g_max = np.asarray(ndi.maximum(canal_verde, lab, index=idx), dtype=float)

    vac = area == 0
    if vac.any():
        d_min[vac] = D_out[filas[vac], cols[vac]]
        g_max[vac] = canal_verde[filas[vac], cols[vac]]

    mask_aisladas = d_min > umbral_dist

    # ── ETAPA 2 : profundidad dentro del verde (solo importa en no-aisladas) ──
    Mg2 = canal_verde > tg_interior
    D_in = ndi.distance_transform_edt(Mg2) if Mg2.any() \
        else np.zeros(canal_verde.shape)
    prof = D_in[filas, cols].astype(float)

    es_interior = (~mask_aisladas) & (prof >= prof_interior)
    es_borde    = (~mask_aisladas) & (prof < prof_interior)

    info = {
        'interior':     es_interior,
        'borde':        es_borde,
        'dist_verde':   d_min,          
        'd_min':        d_min,
        'en_verde':     ~mask_aisladas,
        'prof':         prof,
        'g_max':        g_max,
        'area':         area,
        'umbral_verde': float(umbral_verde),
        'umbral_dist':  float(umbral_dist),
        't_r':          float(t_r),
        'tg_interior':  float(tg_interior),
        'prof_interior': float(prof_interior),
    }
    return mask_aisladas, info


# ──────────────────────────────────────────────────────────────────────────────
# Metricas
# ──────────────────────────────────────────────────────────────────────────────

def confusion(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=bool)
    y_true = np.asarray(y_true, dtype=bool)
    return {
        'TP': int(np.sum( y_pred &  y_true)),
        'TN': int(np.sum(~y_pred & ~y_true)),
        'FP': int(np.sum( y_pred & ~y_true)),
        'FN': int(np.sum(~y_pred &  y_true)),
    }


def metricas_desde_confusion(c):
    TP, TN, FP, FN = c['TP'], c['TN'], c['FP'], c['FN']
    total = TP + TN + FP + FN
    aciertos = TP + TN
    accuracy  = aciertos / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    po = accuracy
    p_real = (TP + FN) / total if total else 0.0
    p_pred = (TP + FP) / total if total else 0.0
    pe = p_real * p_pred + (1 - p_real) * (1 - p_pred)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    return {'total': total, 'aciertos': aciertos, 'accuracy': accuracy,
            'precision': precision, 'recall': recall, 'f1': f1, 'kappa': kappa, **c}


def metricas_completas(c):
    TP, TN, FP, FN = c['TP'], c['TN'], c['FP'], c['FN']
    total = TP + TN + FP + FN
    P = TP + FN

    def _s(a, b):
        return a / b if b else 0.0

    accuracy    = _s(TP + TN, total)
    precision   = _s(TP, TP + FP)
    recall      = _s(TP, TP + FN)
    specificity = _s(TN, TN + FP)
    npv         = _s(TN, TN + FN)
    f1          = _s(2 * precision * recall, precision + recall)
    bal_acc     = (recall + specificity) / 2
    youden      = recall + specificity - 1
    mcc_den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = ((TP * TN - FP * FN) / mcc_den) if mcc_den > 0 else 0.0
    po = accuracy
    p_real = _s(P, total); p_pred = _s(TP + FP, total)
    pe = p_real * p_pred + (1 - p_real) * (1 - p_pred)
    kappa = _s(po - pe, 1 - pe)
    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'n_evaluadas': total,
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'specificity': specificity, 'npv': npv, 'f1': f1,
        'balanced_accuracy': bal_acc, 'mcc': mcc, 'kappa': kappa,
        'fpr': _s(FP, FP + TN), 'fnr': _s(FN, FN + TP), 'fdr': _s(FP, FP + TP),
        'for': _s(FN, FN + TN), 'youden_j': youden, 'prevalence': _s(P, total),
    }


def evaluar(nombre, y_pred, y_true, clase_positiva='aislada', imprimir=True):
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
# Barrido de theta (etapa 1)
# ──────────────────────────────────────────────────────────────────────────────

def barrer_par(tif_path, csv_path, args):
    '''Barre theta (umbral de distancia) sobre un par. Devuelve acumulado o None.'''
    canal_rojo, canal_verde = load_tif_crudo(tif_path)
    positions, clases_raw, y_true = cargar_ground_truth(csv_path)
    if y_true is None:
        return None

    umbrales_dist = args.umbrales_dist if args.umbrales_dist else REJILLA_UMBRAL_DIST

    # Calculamos d_min una sola vez (no depende de theta) y barremos el umbral.
    _, info = clasificar_edt(canal_rojo, canal_verde, positions,
                             umbral_dist=0.0, umbral_verde=args.umbral_verde or 0.0,
                             factor_rojo=args.factor_rojo)
    d_min = info['d_min']
    acum = {}
    for ud in umbrales_dist:
        mask = d_min > ud
        acum[('sop', ud)] = confusion(mask, y_true)
    return acum


def sumar_acumulados(glob_acum, ac):
    for k, c in ac.items():
        if k not in glob_acum:
            glob_acum[k] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for kk in ('TP', 'TN', 'FP', 'FN'):
            glob_acum[k][kk] += c[kk]


def _fmt_param(v):
    if v is None:
        return 'otsu'
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def imprimir_tabla_barrido(acumulado, nombres_params, titulo, ruta_csv=None):
    filas = []
    for params, c in acumulado.items():
        m = metricas_desde_confusion(c)
        filas.append((m['f1'], params, m))
    filas.sort(key=lambda r: r[0], reverse=True)
    best_f1 = filas[0][0]

    if ruta_csv:
        registros = []
        for f1, params, m in filas:
            fila = {nombre: p for nombre, p in zip(nombres_params, params)}
            fila.update({
                'accuracy': round(m['accuracy'], 4), 'precision': round(m['precision'], 4),
                'recall': round(m['recall'], 4), 'f1': round(m['f1'], 4),
                'kappa': round(m['kappa'], 4),
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
# Procesado de un par (evaluacion jerarquica + CSV de detalle)
# ──────────────────────────────────────────────────────────────────────────────

def procesar_par(tif_path, csv_path, args, out_dir=None, imprimir=True):
    if imprimir:
        print(f"\n{'─'*70}")
        print(f"  Imagen : {tif_path}")
        print(f"  CSV    : {csv_path}")

    canal_rojo, canal_verde = load_tif_crudo(tif_path)
    positions, clases_raw, y_true = cargar_ground_truth(csv_path)
    tiene_etiquetas = y_true is not None

    mask, info = clasificar_edt(
        canal_rojo, canal_verde, positions,
        umbral_dist=args.umbral_dist, umbral_verde=(args.umbral_verde or 0.0),
        factor_rojo=args.factor_rojo, tg_interior=args.tg_interior,
        prof_interior=args.prof_interior)

    if imprimir:
        modo = "evaluación jerárquica" if tiene_etiquetas else "inferencia (sin clase)"
        print(f"  Tamaño : {canal_verde.shape[1]} x {canal_verde.shape[0]} px  |  "
              f"Trayectorias: {len(positions)}  |  Modo: {modo}")
        print(f"  Etapa1: t_g={info['umbral_verde']:g} θ={info['umbral_dist']:g}  |  "
              f"Etapa2: t_g={info['tg_interior']:g} p={info['prof_interior']:g}  |  "
              f"no-aisladas={int(np.sum(info['en_verde']))}")

    conf = None
    if tiene_etiquetas:
        y_true_interior = np.array([c in CLASES_INTERIOR for c in clases_raw], dtype=bool)
        y_true_borde    = np.array([c in CLASES_BORDE    for c in clases_raw], dtype=bool)

        if imprimir:
            print(f"\n{'='*50}")
            print("  EVALUACIÓN JERÁRQUICA (2 ETAPAS)")
            print(f"{'='*50}")

        res_aislada = evaluar("Etapa 1: AISLADA vs RESTO", mask, y_true,
                              clase_positiva='aislada', imprimir=imprimir)

        filtro_no_aisladas = y_true_interior | y_true_borde
        if np.any(filtro_no_aisladas):
            res_int_borde = evaluar(
                "Etapa 2: INTERIOR vs BORDE",
                info['interior'][filtro_no_aisladas],
                y_true_interior[filtro_no_aisladas],
                clase_positiva='interior', imprimir=imprimir)
        else:
            res_int_borde = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
            if imprimir:
                print("\n── Etapa 2: sin interior/borde para evaluar ──")

        conf = {
            'aislada':           {k: res_aislada[k]   for k in ('TP', 'TN', 'FP', 'FN')},
            'interior_vs_borde': {k: res_int_borde[k] for k in ('TP', 'TN', 'FP', 'FN')},
        }
    elif imprimir:
        print(f"  Aisladas detectadas: {int(mask.sum())}/{len(positions)}")

    # ── CSV de detalle por trayectoria ────────────────────────────────────────
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
        df_out['V_prof']       = np.round(info['prof'], 2)
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

    return conf


# ──────────────────────────────────────────────────────────────────────────────
# Evaluacion completa por directorio: CSV clasificado + metricas
# ──────────────────────────────────────────────────────────────────────────────

def evaluar_completo_dir(tif_paths, csv_paths, args, out_dir='resultados_EDT'):
    os.makedirs(out_dir, exist_ok=True)

    filas_metricas = []
    glob_ib  = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    glob_ais = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    n_con_etiquetas = 0

    print("\n" + "═" * 74)
    print(f"  EVALUACIÓN COMPLETA EDT (2 etapas)  —  {len(tif_paths)} archivo/s")
    print(f"  Etapa 1: aislada⟺d_min>θ (θ={args.umbral_dist:g}, t_g={args.umbral_verde or 0:g})")
    print(f"  Etapa 2: interior⟺prof≥p (p={args.prof_interior:g}, t_g={args.tg_interior:g})")
    print(f"  Métrica reportada : INTERIOR(+) vs BORDE(−) [aisladas GT excluidas]")
    print(f"  Salida            : {os.path.abspath(out_dir)}")
    print("═" * 74)

    for tif_path, csv_path in zip(tif_paths, csv_paths):
        base = os.path.splitext(os.path.basename(csv_path))[0]
        canal_rojo, canal_verde = load_tif_crudo(tif_path)
        positions, clases_raw, y_true = cargar_ground_truth(csv_path)

        if y_true is None:
            print(f"\n  [AVISO] {base}: sin columna 'clase' — se omite.")
            continue
        n_con_etiquetas += 1

        mask, info = clasificar_edt(
            canal_rojo, canal_verde, positions,
            umbral_dist=args.umbral_dist, umbral_verde=(args.umbral_verde or 0.0),
            factor_rojo=args.factor_rojo, tg_interior=args.tg_interior,
            prof_interior=args.prof_interior)

        y_true_interior = np.array([c in CLASES_INTERIOR for c in clases_raw], dtype=bool)
        y_true_borde    = np.array([c in CLASES_BORDE    for c in clases_raw], dtype=bool)

        # ── Columna de clasificacion (3 categorias) ──
        clasif = np.where(mask, 'Aislada',
                          np.where(info['interior'], 'Interior', 'Borde'))

        df_out = pd.read_csv(csv_path, encoding='utf-8-sig')
        if len(df_out) != len(clasif):
            raise ValueError(f"{base}: desajuste de filas "
                             f"({len(df_out)} vs {len(clasif)}).")
        df_out['clasificacion'] = clasif
        df_out['V_dist_verde']  = np.round(info['dist_verde'], 2)
        df_out['V_prof']        = np.round(info['prof'], 2)
        df_out['V_en_verde']    = info['en_verde'].astype(int)
        ruta_clasif = os.path.join(out_dir, base + '_clasificado_EDT.csv')
        df_out.to_csv(ruta_clasif, index=False, encoding='utf-8-sig')

        m_ais = evaluar(f"[{base}] Etapa 1: AISLADA vs RESTO", mask, y_true,
                        clase_positiva='aislada', imprimir=True)
        for k in ('TP', 'TN', 'FP', 'FN'):
            glob_ais[k] += m_ais[k]

        filtro = y_true_interior | y_true_borde
        conf_ib = confusion(info['interior'][filtro], y_true_interior[filtro])
        for k in ('TP', 'TN', 'FP', 'FN'):
            glob_ib[k] += conf_ib[k]

        m_ib = metricas_completas(conf_ib)
        print(f"   → Interior(+)/Borde(−): "
              f"TP={m_ib['TP']} FP={m_ib['FP']} FN={m_ib['FN']} TN={m_ib['TN']}  |  "
              f"F1={m_ib['f1']:.4f}  Acc={m_ib['accuracy']:.4f}  "
              f"BalAcc={m_ib['balanced_accuracy']:.4f}  Kappa={m_ib['kappa']:.4f}")
        print(f"   CSV clasificado: {ruta_clasif}")

        fila = {
            'archivo': base, 'n_total': len(positions),
            'n_aisladas_gt': int(np.sum(y_true)),
            'umbral_dist': args.umbral_dist,
            'umbral_verde_tg': (args.umbral_verde or 0.0),
            'tg_interior': args.tg_interior, 'prof_interior': args.prof_interior,
        }
        fila.update({k: (round(v, 6) if isinstance(v, float) else v)
                     for k, v in m_ib.items()})
        filas_metricas.append(fila)

    if n_con_etiquetas == 0:
        print("\n  [ERROR] Ningún archivo tenía columna 'clase'.")
        return

    m_glob = metricas_completas(glob_ib)
    fila_glob = {
        'archivo': 'GLOBAL_micro',
        'n_total': sum(f['n_total'] for f in filas_metricas),
        'n_aisladas_gt': sum(f['n_aisladas_gt'] for f in filas_metricas),
        'umbral_dist': args.umbral_dist,
        'umbral_verde_tg': (args.umbral_verde or 0.0),
        'tg_interior': args.tg_interior, 'prof_interior': args.prof_interior,
    }
    fila_glob.update({k: (round(v, 6) if isinstance(v, float) else v)
                      for k, v in m_glob.items()})

    df_met = pd.DataFrame(filas_metricas + [fila_glob])
    ruta_met = os.path.join(out_dir, 'metricas_EDT.csv')
    df_met.to_csv(ruta_met, index=False, encoding='utf-8-sig')

    m_ais_glob = metricas_desde_confusion(glob_ais)
    sep = "═" * 74
    print(f"\n{sep}")
    print(f"  RESUMEN GLOBAL (micro)  —  {n_con_etiquetas} archivo/s")
    print(f"{sep}")
    print(f"  Etapa 1 AISLADA : Acc={m_ais_glob['accuracy']:.4f}  "
          f"Prec={m_ais_glob['precision']:.4f}  Rec={m_ais_glob['recall']:.4f}  "
          f"F1={m_ais_glob['f1']:.4f}  kappa={m_ais_glob['kappa']:.4f}")
    print(f"  Etapa 2 INT/BOR : TP={m_glob['TP']} TN={m_glob['TN']} "
          f"FP={m_glob['FP']} FN={m_glob['FN']}")
    print(f"                    Acc={m_glob['accuracy']:.4f}  "
          f"BalAcc={m_glob['balanced_accuracy']:.4f}  F1={m_glob['f1']:.4f}  "
          f"kappa={m_glob['kappa']:.4f}  MCC={m_glob['mcc']:.4f}")
    print(f"  Métricas -> {ruta_met}")
    print(f"{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Filtrado -> <base>_filtrado.csv  (X, Y, clase, Filtradas, V_interior, V_borde)
# ──────────────────────────────────────────────────────────────────────────────

def filtrar_particulas(tif_path, csv_path, args, ruta_salida=None, imprimir=True):
    canal_rojo, canal_verde = load_tif_crudo(tif_path)
    positions, _, _ = cargar_ground_truth(csv_path)

    mask, info = clasificar_edt(
        canal_rojo, canal_verde, positions,
        umbral_dist=args.umbral_dist, umbral_verde=(args.umbral_verde or 0.0),
        factor_rojo=args.factor_rojo, tg_interior=args.tg_interior,
        prof_interior=args.prof_interior)

    filtradas = np.where(mask, ETIQUETA_FILTRADA, '')

    df_orig = pd.read_csv(csv_path, encoding='utf-8-sig')
    mapa = {c.strip().lower(): c for c in df_orig.columns}
    if 'x' not in mapa or 'y' not in mapa:
        raise ValueError("El CSV debe contener columnas X e Y.")
    if len(df_orig) != len(positions):
        raise ValueError("Desajuste de filas entre CSV y posiciones.")

    col_x, col_y = mapa['x'], mapa['y']
    col_clase = mapa.get('clase')

    df_out = pd.DataFrame()
    df_out['X'] = df_orig[col_x].to_numpy()
    df_out['Y'] = df_orig[col_y].to_numpy()
    df_out['clase'] = df_orig[col_clase].to_numpy() if col_clase else ''
    df_out['Filtradas'] = filtradas
    df_out['V_interior'] = info['interior'].astype(int)
    df_out['V_borde']    = info['borde'].astype(int)

    if ruta_salida is None:
        ruta_salida = os.path.splitext(csv_path)[0] + "_filtrado.csv"
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(ruta_salida, index=False, encoding='utf-8-sig')

    if imprimir:
        n_aisl = int(mask.sum()); n_tot = len(positions)
        print(f"\n{'─'*70}")
        print(f"  Filtrado (EDT 2 etapas)")
        print(f"  Imagen : {tif_path}")
        print(f"  Aisladas marcadas: {n_aisl}/{n_tot} "
              f"({100.0 * n_aisl / max(1, n_tot):.1f}%)")
        if col_clase is None:
            print("  [AVISO] El CSV no tenía columna 'clase'; se escribe vacía.")
        print(f"  Guardado : {ruta_salida}")
    return ruta_salida


# ──────────────────────────────────────────────────────────────────────────────
# Emparejar .tif con .csv en un directorio
# ──────────────────────────────────────────────────────────────────────────────

def buscar_pares(directorio):
    tif_paths, csv_paths = [], []
    for tif in sorted(glob.glob(os.path.join(directorio, "*.tif"))):
        base = os.path.splitext(os.path.basename(tif))[0]
        if base.endswith("_deteccion") or base.endswith("_filtrado") \
                or base.endswith("_clasificado_EDT"):
            continue
        csv = os.path.splitext(tif)[0] + ".csv"
        if os.path.isfile(csv):
            tif_paths.append(tif)
            csv_paths.append(csv)
    return tif_paths, csv_paths


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACION DE MODOS DE EJECUCION Y BUCLE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def construir_parser():
    p = argparse.ArgumentParser(
        description="Clasificador EDT en 2 etapas (aislada; interior/borde).")
    p.add_argument("--tif", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--dir", default=None)

    # Parametros del metodo
    p.add_argument("--umbral-dist", type=float, default=UMBRAL_DIST_DEF,
                   help="theta (etapa 1). Por defecto 0.")
    p.add_argument("--umbral-verde", type=float, default=UMBRAL_VERDE_DEF,
                   help="t_g de la etapa 1 (mascara de soporte). Por defecto 0.")
    p.add_argument("--factor-rojo", type=float, default=FACTOR_ROJO_DEF,
                   help="t_r = factor*Otsu(rojo) para el watershed. Por defecto 0.25.")
    p.add_argument("--tg-interior", type=float, default=TG_INTERIOR_DEF,
                   help="t_g de la etapa 2 (profundidad). Por defecto 12.")
    p.add_argument("--prof-interior", type=float, default=PROF_INTERIOR_DEF,
                   help="p_prof: interior <=> prof>=p_prof. Por defecto 4.")

    # Modos
    p.add_argument("--evaluar-completo", action="store_true",
                   help="CSV clasificado + metricas_EDT.csv en --out-dir.")
    p.add_argument("--filtrar", action="store_true",
                   help="Genera <base>_filtrado.csv (X, Y, clase, Filtradas, ...).")
    p.add_argument("--barrer", action="store_true",
                   help="Barrido de theta (etapa 1) sobre archivos con clase.")
    p.add_argument("--guardar-csv", default=None,
                   help="Ruta del CSV de detalle por trayectoria (modo par).")
    p.add_argument("--guardar-filtrado", default=None)
    p.add_argument("--guardar-barrido", default=None)
    p.add_argument("--out-dir", default="resultados_EDT")
    p.add_argument("--umbrales-dist", nargs="+", type=float, default=None,
                   help="Rejilla de theta para el barrido.")
    return p


def main(argv=None):
    args = construir_parser().parse_args(argv)

    if not args.dir and not (args.tif and args.csv):
        print("[ERROR] Indica --dir DIRECTORIO, o bien --tif y --csv.")
        sys.exit(1)

    # ── MODO DIRECTORIO ───────────────────────────────────────────────────────
    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"[ERROR] No es un directorio: {args.dir}"); sys.exit(1)
        tif_paths, csv_paths = buscar_pares(args.dir)
        if not tif_paths:
            print("[ERROR] No se encontró ningún par .csv + .tif."); sys.exit(1)

        print(f"\n  Directorio        : {os.path.abspath(args.dir)}")
        print(f"  Pares encontrados : {len(tif_paths)}")

        if args.evaluar_completo:
            evaluar_completo_dir(tif_paths, csv_paths, args, out_dir=args.out_dir)
            return
        if args.filtrar:
            for tif_path, csv_path in zip(tif_paths, csv_paths):
                try:
                    filtrar_particulas(tif_path, csv_path, args)
                except Exception as e:
                    print(f"  [ERROR] {os.path.basename(csv_path)}: {e}")
            return
        if args.barrer:
            glob_acum, usados = {}, 0
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
                print("\n  [ERROR] Ningún archivo tenía columna 'clase'."); sys.exit(1)
            print(f"\n  Barrido global sobre {usados} imagen/es.")
            imprimir_tabla_barrido(glob_acum, ["mascVerde", "umbDist"],
                                   f"BARRIDO GLOBAL θ (etapa 1) — {usados} imágenes",
                                   ruta_csv=args.guardar_barrido)
            return

        # Evaluacion jerarquica con parametros fijos por par
        glob_acum = {
            'aislada':           {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
            'interior_vs_borde': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
        }
        con_etiquetas = 0
        for tif_path, csv_path in zip(tif_paths, csv_paths):
            try:
                conf = procesar_par(tif_path, csv_path, args, out_dir=None)
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
            print(f"  Etapa 1 (Aisladas)   : Acc {m1['accuracy']:.1%}  "
                  f"Prec {m1['precision']:.1%}  Rec {m1['recall']:.1%}  "
                  f"F1 {m1['f1']:.1%}  kappa {m1['kappa']:.2f}")
            m2 = metricas_desde_confusion(glob_acum['interior_vs_borde'])
            if m2['total'] > 0:
                print(f"  Etapa 2 (Int vs Bor) : Acc {m2['accuracy']:.1%}  "
                      f"Prec {m2['precision']:.1%}  Rec {m2['recall']:.1%}  "
                      f"F1 {m2['f1']:.1%}  kappa {m2['kappa']:.2f}")
        else:
            print("  Ningún archivo tenía columna 'clase'.")
        print(f"{'═'*70}\n")
        return

    # ── MODO ARCHIVO INDIVIDUAL ───────────────────────────────────────────────
    for path in (args.tif, args.csv):
        if not os.path.isfile(path):
            print(f"[ERROR] No se encuentra: {path}"); sys.exit(1)

    if args.evaluar_completo:
        evaluar_completo_dir([args.tif], [args.csv], args, out_dir=args.out_dir)
        return
    if args.filtrar:
        filtrar_particulas(args.tif, args.csv, args, ruta_salida=args.guardar_filtrado)
        return
    if args.barrer:
        ac = barrer_par(args.tif, args.csv, args)
        if ac is None:
            print("[ERROR] El CSV no tiene columna 'clase'."); sys.exit(1)
        imprimir_tabla_barrido(ac, ["mascVerde", "umbDist"],
                               "BARRIDO θ (etapa 1)", ruta_csv=args.guardar_barrido)
        return

    procesar_par(args.tif, args.csv, args, out_dir=None)


if __name__ == "__main__":
    main()