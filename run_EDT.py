#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_detectar.py
================================================================================
Lanzador de detectar_aisladas.py al estilo "configurar y descomentar".

NO hace falta usar la terminal con flags: configuras las variables de abajo,
eliges el MODO descomentando UNA línea, y ejecutas:

    python run_detectar.py

Internamente construye los argumentos y llama a detectar_aisladas.main(...),
así que reutiliza exactamente la misma lógica que la versión de terminal.
================================================================================
"""

import detectar_aisladas   # debe estar en el mismo directorio que este archivo


# ══════════════════════════════════════════════════════════════════════════════
# 1) ELIGE EL MODO  (descomenta SOLO una línea)
# ══════════════════════════════════════════════════════════════════════════════

# MODO = "evaluar"       # Modo 1: evaluar el método con parámetros fijos (un par)
MODO = "evaluar_dir"   # Modo 2: evaluar con parámetros fijos en un directorio
# MODO = "barrer"        # Modo 3: barrido de parámetros (un par)
# MODO = "barrer_dir"    # Modo 4: barrido GLOBAL acumulado sobre un directorio
# MODO = "filtrar"       # Modo 5: filtra y genera <base>_filtrado.csv (un par)
# MODO = "filtrar_dir"   # Modo 6: igual que filtrar, pero sobre un directorio
# MODO = "pintar_edt"    # Modo 7: pinta el campo de la EDT para verlo (un par)
# MODO = "ver_mascaras"  # Modo 8: diagnóstico de las máscaras Mg/Mr (depurar)


# ══════════════════════════════════════════════════════════════════════════════
# 2) RUTAS
# ══════════════════════════════════════════════════════════════════════════════

# --- Para los modos de UN par (evaluar, barrer, filtrar) ---
TIF = "../datos/definitivos/SUb_02_8.tif"
CSV = "../datos/definitivos/SUb_02_8.csv"

# --- Para los modos de DIRECTORIO (evaluar_dir, barrer_dir, filtrar_dir) ---
DIR = "../datos/definitivos"


# ══════════════════════════════════════════════════════════════════════════════
# 3) PARÁMETROS FIJOS DEL MÉTODO  (evaluar, evaluar_dir, filtrar, inferencia)
# ══════════════════════════════════════════════════════════════════════════════

UMBRAL_VERDE = None     # umbral de binarización del verde. None = Otsu, o un float (p.ej. 0.05)
UMBRAL_ROJO  = None  # umbral del rojo (solo conectividad). None = Otsu
UMBRAL_DIST  = 4       # umbral de distancia θ (px): aislada si d_g > θ (he puesto el mejor encontrado hasta ahora)
CONECTIVIDAD = True      # True = una isla roja que toca el verde NO es aislada

# Segmentación del verde (para que Mg cubra bien el verde difuso)
UMBRAL_VERDE_BAJO = None # histéresis: umbral bajo (None = sin histéresis). P.ej. 0.05
VERDE_SIGMA       = 0.0  # suavizado gaussiano del verde antes de umbralar (0 = off)
VERDE_CIERRE      = 0    # cierre morfológico de Mg en iteraciones (0 = off)
VERDE_RELLENAR    = False # rellenar huecos interiores de Mg


# ══════════════════════════════════════════════════════════════════════════════
# 4) REJILLAS DE BARRIDO  (barrer, barrer_dir)
#    Deja una lista vacía []  para usar la rejilla por defecto del script.
# ══════════════════════════════════════════════════════════════════════════════

UMBRALES_VERDE = []                      # [] = por defecto [Otsu]
UMBRALES_DIST  = [4,8,12,16,20]  # umbrales de distancia θ a probar


# ══════════════════════════════════════════════════════════════════════════════
# 5) FILTRADO  (filtrar, filtrar_dir)
# ══════════════════════════════════════════════════════════════════════════════

GUARDAR_FILTRADO = None   # ruta del CSV filtrado en modo de UN par.
                          # None -> <base>_filtrado.csv junto al CSV de entrada.


# ══════════════════════════════════════════════════════════════════════════════
# 6) SALIDAS  (deja en None / False lo que no quieras generar)
# ══════════════════════════════════════════════════════════════════════════════

GUARDAR_CSV     = "resultado.csv"     # detalle por partícula (modos de un par)
GUARDAR_FIGURA  = "deteccion.png"     # overlay TP/FP/FN/TN + banda θ (modo evaluar)
MOSTRAR_FALLOS  = True                # listar FP/FN con coordenadas (modo evaluar)
GUARDAR_BARRIDO = "barrido.csv"       # tabla del barrido (modos barrer)


# ══════════════════════════════════════════════════════════════════════════════
# 7) PINTAR EDT  (modo pintar_edt)
# ══════════════════════════════════════════════════════════════════════════════

SOBRE_EDT    = "verde"          # "verde" (distancia a la traza) o "rojo" (blobs al fondo)
EDT_DISCRETO = True           # False = mapa continuo con barra; True = tramos de color
CORTES_EDT   = [1, 5, 10]       # cortes en px del modo discreto (el valor 0 va aparte)
COLORES_EDT  = []               # [] = paleta verde→naranja→rojo automática; o lista de hex
GUARDAR_EDT  = "campo_edt.png"  # ruta de la imagen del campo EDT
GUARDAR_MASCARAS = "mascaras.png"  # diagnóstico de Mg/Mr (modo ver_mascaras)


# ══════════════════════════════════════════════════════════════════════════════
# A PARTIR DE AQUÍ NO HACE FALTA TOCAR NADA: se construyen los argumentos
# ══════════════════════════════════════════════════════════════════════════════

def _añadir_lista(argv, flag, valores):
    '''Añade '--flag v1 v2 ...' solo si la lista tiene elementos.'''
    if valores:
        argv.append(flag)
        argv.extend(str(v) for v in valores)


def construir_argv():
    '''Traduce las variables de configuración a la lista de flags del CLI.'''
    argv = []

    # ── Parámetros fijos comunes ──────────────────────────────────────────────
    argv += ["--umbral-dist", str(UMBRAL_DIST)]
    if UMBRAL_VERDE is not None:
        argv += ["--umbral-verde", str(UMBRAL_VERDE)]
    if UMBRAL_ROJO is not None:
        argv += ["--umbral-rojo", str(UMBRAL_ROJO)]
    if not CONECTIVIDAD:
        argv += ["--sin-conectividad"]
    # Segmentación del verde
    if UMBRAL_VERDE_BAJO is not None:
        argv += ["--umbral-verde-bajo", str(UMBRAL_VERDE_BAJO)]
    if VERDE_SIGMA and VERDE_SIGMA > 0:
        argv += ["--verde-sigma", str(VERDE_SIGMA)]
    if VERDE_CIERRE and VERDE_CIERRE > 0:
        argv += ["--verde-cierre", str(VERDE_CIERRE)]
    if VERDE_RELLENAR:
        argv += ["--verde-rellenar"]

    # ── Fuente de datos según el modo ─────────────────────────────────────────
    if MODO in ("evaluar", "barrer", "filtrar", "pintar_edt", "ver_mascaras"):
        argv += ["--tif", TIF, "--csv", CSV]
    elif MODO in ("evaluar_dir", "barrer_dir", "filtrar_dir"):
        argv += ["--dir", DIR]
    else:
        raise ValueError(f"MODO desconocido: {MODO!r}")

    # ── Opciones específicas de cada modo ─────────────────────────────────────
    if MODO in ("barrer", "barrer_dir"):
        argv += ["--barrer"]
        _añadir_lista(argv, "--umbrales-verde", UMBRALES_VERDE)
        _añadir_lista(argv, "--umbrales-dist",  UMBRALES_DIST)
        if GUARDAR_BARRIDO:
            argv += ["--guardar-barrido", GUARDAR_BARRIDO]
    elif MODO in ("filtrar", "filtrar_dir"):
        argv += ["--filtrar"]
        if MODO == "filtrar" and GUARDAR_FILTRADO:
            argv += ["--guardar-filtrado", GUARDAR_FILTRADO]
    elif MODO == "pintar_edt":
        argv += ["--guardar-edt", GUARDAR_EDT, "--edt-sobre", SOBRE_EDT]
        if EDT_DISCRETO:
            _añadir_lista(argv, "--edt-cortes",  CORTES_EDT)
            _añadir_lista(argv, "--edt-colores", COLORES_EDT)
    elif MODO == "ver_mascaras":
        argv += ["--guardar-mascaras", GUARDAR_MASCARAS]
    else:
        # evaluar / evaluar_dir
        if GUARDAR_CSV:
            argv += ["--guardar-csv", GUARDAR_CSV]
        if MODO == "evaluar":
            if GUARDAR_FIGURA:
                argv += ["--guardar-figura", GUARDAR_FIGURA]
            if MOSTRAR_FALLOS:
                argv += ["--mostrar-fallos"]

    return argv


if __name__ == "__main__":
    argv = construir_argv()
    print(f"MODO = {MODO}")
    print("Equivale en terminal a:\n   python detectar_aisladas.py " + " ".join(argv) + "\n")
    detectar_aisladas.main(argv)