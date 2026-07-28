#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_EDT.py
================================================================================
Lanzador de clasificacion_EDT.py al estilo "configurar y descomentar".

NO hace falta escribir flags en la terminal: configuras las variables de abajo,
eliges el MODO descomentando UNA linea, y ejecutas:

    python run_EDT.py

Internamente construye los argumentos y llama a clasificacion_EDT.main(...), asi
que reutiliza exactamente la misma logica que la version de terminal.

--------------------------------------------------------------------------------
El clasificador tiene DOS etapas:
    Etapa 1 (aislada / no aislada) : aislada <=> d_min > UMBRAL_DIST
                                     (d_min = distancia del territorio rojo al
                                      verde; territorio por watershed)
    Etapa 2 (interior / borde)     : interior <=> prof >= PROF_INTERIOR
                                     (prof = profundidad del centroide dentro de
                                      la mascara verde {I_g > TG_INTERIOR})
================================================================================
"""

import clasificacion_EDT   # debe estar en el mismo directorio que este archivo


# ══════════════════════════════════════════════════════════════════════════════
# 1) ELIGE EL MODO  (descomenta SOLO una linea)
# ══════════════════════════════════════════════════════════════════════════════

MODO = "evaluar"          # Modo 1: evaluar (un par) con parametros fijos + CSV detalle
# MODO = "evaluar_dir"      # Modo 2: evaluar jerarquico sobre un directorio
# MODO = "evaluar_completo"   # Modo 2b: clasifica + <base>_clasificado_EDT.csv y
                            #          metricas_EDT.csv en OUT_DIR_EVAL
# MODO = "barrer"           # Modo 3: barrido de theta (etapa 1) sobre un par
# MODO = "barrer_dir"       # Modo 4: barrido GLOBAL de theta sobre un directorio
# MODO = "filtrar"          # Modo 5: <base>_filtrado.csv (un par) — pre-filtro CNN
# MODO = "filtrar_dir"      # Modo 6: igual que filtrar, sobre un directorio


# ══════════════════════════════════════════════════════════════════════════════
# 2) RUTAS
# ══════════════════════════════════════════════════════════════════════════════

# --- Modos de UN par (evaluar, barrer, filtrar) ---
TIF = "../datos/imagenes_nuevas/test/SUboligo_02_3_merged.tif"
CSV = "../datos/imagenes_nuevas/test/SUboligo_02_3_merged_anotaciones.csv"

# --- Modos de DIRECTORIO (evaluar_dir, barrer_dir, filtrar_dir, evaluar_completo) ---
DIR = "../datos/definitivos"


# ══════════════════════════════════════════════════════════════════════════════
# 3) PARAMETROS DEL METODO
#    Los valores por defecto son los fijados empiricamente; no deberia hacer
#    falta tocarlos.
# ══════════════════════════════════════════════════════════════════════════════

# --- Etapa 1: aisladas ---
UMBRAL_DIST  = 0.0    # theta (px). aislada <=> d_min > theta.  Optimo empirico: 0
UMBRAL_VERDE = 0.0    # t_g de la mascara de soporte Mg = {I_g > t_g}.  Optimo: 0
FACTOR_ROJO  = 0.25   # t_r = FACTOR_ROJO * Otsu(rojo>0) para el watershed

# --- Etapa 2: interior / borde ---
TG_INTERIOR   = 12.0  # t_g de la mascara verde para medir la profundidad
PROF_INTERIOR = 4.0   # interior <=> prof >= PROF_INTERIOR (px)


# ══════════════════════════════════════════════════════════════════════════════
# 4) REJILLA DE BARRIDO  (barrer, barrer_dir)
#    Barre theta de la etapa 1. Lista vacia [] = rejilla por defecto del script.
# ══════════════════════════════════════════════════════════════════════════════

UMBRALES_DIST = [0, 1, 2, 3, 5]


# ══════════════════════════════════════════════════════════════════════════════
# 5) SALIDAS
# ══════════════════════════════════════════════════════════════════════════════

GUARDAR_CSV      = "resultado.csv"    # detalle por trayectoria (modos de un par)
GUARDAR_FILTRADO = None               # None -> <base>_filtrado.csv junto al CSV
GUARDAR_BARRIDO  = "barrido.csv"      # tabla del barrido (modos barrer)
OUT_DIR_EVAL     = "resultados_EDT"   # CSVs clasificados + metricas



def _añadir_lista(argv, flag, valores):
    if valores:
        argv.append(flag)
        argv.extend(str(v) for v in valores)


def construir_argv():
    argv = []

    # ── Parametros del metodo (comunes a todos los modos) ─────────────────────
    argv += ["--umbral-dist",   str(UMBRAL_DIST)]
    argv += ["--umbral-verde",  str(UMBRAL_VERDE)]
    argv += ["--factor-rojo",   str(FACTOR_ROJO)]
    argv += ["--tg-interior",   str(TG_INTERIOR)]
    argv += ["--prof-interior", str(PROF_INTERIOR)]

    # ── Fuente de datos segun el modo ─────────────────────────────────────────
    if MODO in ("evaluar", "barrer", "filtrar"):
        argv += ["--tif", TIF, "--csv", CSV]
    elif MODO in ("evaluar_dir", "barrer_dir", "filtrar_dir", "evaluar_completo"):
        argv += ["--dir", DIR]
    else:
        raise ValueError(f"MODO desconocido: {MODO!r}")

    # ── Opciones especificas ──────────────────────────────────────────────────
    if MODO in ("barrer", "barrer_dir"):
        argv += ["--barrer"]
        _añadir_lista(argv, "--umbrales-dist", UMBRALES_DIST)
        if GUARDAR_BARRIDO:
            argv += ["--guardar-barrido", GUARDAR_BARRIDO]
    elif MODO in ("filtrar", "filtrar_dir"):
        argv += ["--filtrar"]
        if MODO == "filtrar" and GUARDAR_FILTRADO:
            argv += ["--guardar-filtrado", GUARDAR_FILTRADO]
    elif MODO == "evaluar_completo":
        argv += ["--evaluar-completo", "--out-dir", OUT_DIR_EVAL]
    else:  # evaluar / evaluar_dir
        if GUARDAR_CSV and MODO == "evaluar":
            argv += ["--guardar-csv", GUARDAR_CSV]

    return argv


if __name__ == "__main__":
    argv = construir_argv()
    print(f"MODO = {MODO}")
    print("Equivale en terminal a:\n   python clasificacion_EDT.py "
          + " ".join(argv) + "\n")
    clasificacion_EDT.main(argv)