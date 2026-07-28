# Clasificación de trayectorias estaticas en microscopia de fluorescencia de dos canales

Este repositorio implementa la clasificacion de trayectorias estaticas en imágenes TIFF de
microscopía de fluorescencia de dos canales (canal rojo: trayectorias estáticas; canal verde:
taryectorias elípticas y no homogéneas). Cada trayectoria estática se asigna a una de las tres categorias:

- Interior: trayectoria situada en el interior de la region verde.
- Borde: trayectoria situada en el borde de la region verde.
- Aislada: trayectoria sin solapamiento con el canal verde.

El proyecto contiene dos aproximaciones complementarias:

1. Un clasificador clásico basado en la transformada de distancia euclidea (EDT), en dos etapas.
2. Un clasificador basado en Deep Learning (CNN) con postprocesado por ensemble.

El clasificador EDT cumple un doble papel: es un clasificador clásico independiente y, además, actua
como pre-filtro de aisladas (Etapa 1) antes de la CNN.


## Organización de módulos

El código está organizado de forma que los dos módulos sean independientes, con un único módulo
comun de utilidades:

- funciones_auxiliares.py es el módulo común. Contiene la carga de imágenes y CSV, la extracción de
  patches, el Dataset de entrenamiento, la detección de aisladas de la Etapa 1
  (detectar_aisladas_EDT) y la construcción de territorios watershed (construir_territorios). El
  import de PyTorch es opcional: solo se necesita para el Dataset de entrenamiento, de modo que el
  método clasico puede usar este módulo sin PyTorch instalado.
- clasificacion_EDT.py es el clasificador clásico.
- clasificacion.py y entrenamiento.py forman parte del modelo de la CNN.


## Listado de archivos

- funciones_auxiliares.py    Módulo común: carga de datos, patches, Dataset, detección de aisladas.
- modelo_CNN.py              Arquitectura de la CNN (MitochondriaContextCNN).
- entrenamiento.py           Entrenamiento de la CNN, validacion, early stopping y logs.
- clasificacion.py           Inferencia con la CNN entrenada (pre-filtro de aisladas por Etapa 1 EDT).
- clasificacion_EDT.py       Clasificador clasico EDT en dos etapas (baseline y pre-filtro).
- run_EDT.py                 Lanzador de clasificacion_EDT.py al estilo "configurar y descomentar".
- promediar_clasificacion.py Ensemble por soft voting sobre varios runs de inferencia.
- imagenes_clasificado.py    Visualizacion de resultados sobre la imagen y calculo de metricas por imagen.
- representaciones.py        Figuras SVG a partir de los runs de entrenamiento.
- data_augmentation.py       Aumento de datos (imagenes y CSV) con albumentations.
- anotador_particulas.py     Herramienta interactiva de anotacion manual.


## Requisitos

- Python 3.8 o superior.
- PyTorch (con CUDA si se dispone de GPU). Solo necesario para el pipeline de la CNN.
- numpy, scipy, scikit-image, scikit-learn, pandas, matplotlib, tifffile, opencv-python.
- albumentations (solo para data_augmentation.py).

Instalacion de dependencias (ejemplo con CUDA 11.8):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-image scikit-learn pandas matplotlib tifffile opencv-python albumentations openpyxl
```

Para ejecutar unicamente el clasificador clasico (clasificacion_EDT.py) no hace falta PyTorch:

```
pip install numpy scipy scikit-image scikit-learn pandas matplotlib tifffile
```


## Formato de los datos

Imagen TIFF de entrada:

- Dos canales. Canal 0: rojo (trayectorias estaticas). Canal 1: verde (trazas).
- El fondo del canal verde es exactamente cero. Esta propiedad es la que permite usar la mascara de
  soporte M_g = {I_g > 0} sin ningun umbral libre.

CSV de anotaciones de entrada. Columnas minimas:

- X, Y: coordenadas del centroide de cada trayectoria.
- clase (opcional): interior, borde o aislada.

El comportamiento depende de si el CSV incluye la columna de clase:

- Con columna de clase: modo evaluacion. Las aisladas se excluyen por etiqueta y el resto se
  compara contra las etiquetas reales para calcular metricas.
- Sin columna de clase: modo inferencia. Las aisladas se detectan con la Etapa 1 del EDT.


## Flujo de trabajo completo

El orden habitual de ejecucion es el siguiente:

```
1. Anotacion
   python anotador_particulas.py
   Genera un CSV de anotaciones por imagen.

2. Aumento de datos (opcional, si hay pocas imagenes de entrenamiento)
   python data_augmentation.py
   Genera imagenes y CSV aumentados.

3. Baseline clasico y/o pre-filtro de aisladas
   python run_EDT.py
   Segun el modo: evalua el EDT o genera CSVs pre-filtrados para la CNN.

4. Entrenamiento de la CNN
   CUDA_VISIBLE_DEVICES='0' python3 entrenamiento.py
   Genera best_mito_classifier.pth, curvas y logs.

5. Inferencia con la CNN (repetir varias veces para el ensemble)
   python clasificacion.py
   Genera un CSV <nombre>_clasificado.csv por imagen.

6. Ensemble
   python promediar_clasificacion.py
   Promedia las probabilidades de los distintos runs.

7. Visualizacion y metricas por imagen
   python imagenes_clasificado.py
   Genera PNG, TIFF y CSV de metricas.
```


## Ejecucion detallada de cada script

Todos los lanzadores siguen el patron "configurar variables en la parte superior del bloque
principal, (des)comentar el modo si procede, y ejecutar el script". No se usan flags de terminal,
salvo en promediar_clasificacion.py y representaciones.py, que ademas aceptan argumentos opcionales.


### anotador_particulas.py

Herramienta interactiva para anotar manualmente las trayectorias. Deteccion automatica de puntos
sobre el canal rojo, sliders de contraste, y guardado en CSV.

Configuracion en la parte superior del archivo:

```
IMAGE_PATH          = "../datos/imagenes_nuevas/SUboligo_02_3.tif"  # imagen a anotar
OUTPUT_CSV          = None            # None -> se construye a partir de la imagen
ANNOT_DIR           = "../datos/definitivos"   # carpeta donde guardar/buscar los CSV
LOAD_EXISTING       = True            # reanuda anotaciones previas si existen
DETECTAR_AUTOMATICO = True            # detecta y precarga puntos al abrir la imagen
METODO_DETECCION    = "maximos"       # "maximos" (picos locales) o "centroides" (blobs)
DET_SIGMA           = 1.0             # suavizado gaussiano previo (px); 0 = sin suavizar
DET_UMBRAL          = None            # None -> Otsu automatico sobre el canal rojo
CLASSES             = ["Borde", "Interior", "Aislada"]
EXPORT_INT          = True            # X, Y como enteros (coherente con patches y EDT)
```

Ejecucion:

```
python anotador_particulas.py
```

Salida: un CSV por imagen con las columnas X, Y y la clase asignada.


### data_augmentation.py

Genera versiones aumentadas de cada par imagen mas CSV (rotaciones, flips y demas transformaciones
definidas en el diccionario TRANSFORMS), transformando de forma coherente las coordenadas de las
trayectorias.

Configuracion en la parte superior del archivo:

```
INPUT_DIR  = "../datos/definitivos/training"           # imagenes y CSV originales
OUTPUT_DIR = "../datos/definitivos/data_augmentation"  # salida aumentada
IMAGE_EXT  = ".tif"
```

El script no usa un bloque principal: al ejecutarlo recorre INPUT_DIR, empareja cada imagen con su
CSV del mismo nombre y escribe los resultados en OUTPUT_DIR.

Ejecucion:

```
python data_augmentation.py
```

Salida: para cada imagen de entrada, un TIFF y un CSV por transformacion, con nombre
<base>_<transformacion>.tif y <base>_<transformacion>.csv.


### clasificacion_EDT.py y run_EDT.py

Clasificador clasico en dos etapas:

- Etapa 1 (aislada / no aislada): aislada si d_min > UMBRAL_DIST, donde d_min es la distancia minima
  al verde sobre el territorio watershed del rojo de cada trayectoria.
- Etapa 2 (interior / borde): interior si la profundidad del centroide dentro de la mascara verde
  {I_g > TG_INTERIOR} es mayor o igual que PROF_INTERIOR.

La forma recomendada de lanzarlo es run_EDT.py, que construye los argumentos y llama internamente a
clasificacion_EDT.main(...).

Configuracion en run_EDT.py. Primero se elige el modo descomentando una sola linea:

```
MODO = "evaluar"          # evalua un par (TIF + CSV) y guarda un CSV de detalle
# MODO = "evaluar_dir"      # evalua todos los pares de un directorio
# MODO = "evaluar_completo" # clasifica y guarda <base>_clasificado_EDT.csv y metricas_EDT.csv
# MODO = "barrer"           # barrido de theta (Etapa 1) sobre un par
# MODO = "barrer_dir"       # barrido global de theta sobre un directorio
# MODO = "filtrar"          # genera <base>_filtrado.csv (un par): pre-filtro para la CNN
# MODO = "filtrar_dir"      # igual que filtrar, sobre un directorio
```

Rutas y parametros:

```
# Modos de un par
TIF = "../datos/imagenes_nuevas/test/SUboligo_02_3_merged.tif"
CSV = "../datos/imagenes_nuevas/test/SUboligo_02_3_merged_anotaciones.csv"

# Modos de directorio
DIR = "../datos/definitivos"

# Etapa 1 (aisladas). Los valores por defecto son los optimos empiricos.
UMBRAL_DIST  = 0.0    # theta (px): aislada si d_min > theta
UMBRAL_VERDE = 0.0    # t_g de la mascara de soporte M_g = {I_g > t_g}
FACTOR_ROJO  = 0.25   # t_r = FACTOR_ROJO * Otsu(rojo>0) para el watershed

# Etapa 2 (interior / borde)
TG_INTERIOR   = 12.0  # t_g de la mascara verde para medir la profundidad
PROF_INTERIOR = 4.0   # interior si prof >= PROF_INTERIOR (px)

# Salidas
GUARDAR_CSV  = "resultado.csv"    # detalle por trayectoria (modos de un par)
OUT_DIR_EVAL = "resultados_EDT"   # CSVs clasificados y metricas (evaluar_completo)
```

Ejecucion:

```
python run_EDT.py
```

Nota sobre el pre-filtro: el modo filtrar / filtrar_dir escribe un CSV <base>_filtrado.csv con las
aisladas eliminadas, que puede usarse como entrada de la CNN. Como este pre-filtro alimenta a la
red, interesa maximizar el recall de las no aisladas para no descartar datos validos.


### entrenamiento.py

Entrena la CNN. Excluye automaticamente las trayectorias etiquetadas como aisladas y divide el
conjunto en entrenamiento y validacion. Incluye early stopping opcional sobre el AUC de validacion,
reduccion del learning rate en meseta y guardado del mejor modelo por AUC de validacion.

Ejecucion en GPU. Se indica el identificador de la GPU con CUDA_VISIBLE_DEVICES:

```
CUDA_VISIBLE_DEVICES='0' python3 entrenamiento.py
```

Otros ejemplos de seleccion de dispositivo:

```
CUDA_VISIBLE_DEVICES='1' python3 entrenamiento.py     # usar la GPU 1
CUDA_VISIBLE_DEVICES='0,1' python3 entrenamiento.py   # exponer dos GPUs
CUDA_VISIBLE_DEVICES='' python3 entrenamiento.py      # forzar CPU (mucho mas lento)
```

Configuracion en el bloque principal (parte final del archivo). Hay tres formas de indicar los
datos; se deja activa una y se comentan las demas:

```
# Opcion A: una sola imagen
# TIF_PATH = "datos/SUb_01_2_merged.tif"
# CSV_PATH = "datos/SUb_01_2_datos_training.csv"

# Opcion B: listas manuales
# TIF_PATH = ["datos/img1.tif", "datos/img2.tif"]
# CSV_PATH = ["datos/img1.csv", "datos/img2.csv"]

# Opcion C: directorio completo (util con datos aumentados)
TIF_PATH, CSV_PATH = load_pairs_from_dir("../data_augmentation_clase")

SAVE_DIR   = "resultados_entrenamiento"   # curvas, Excel y CSV de metadata
MODEL_PATH = 'best_mito_classifier.pth'

EPOCAS                   = 300
USAR_EARLY_STOPPING      = True
EARLY_STOPPING_PATIENCE  = 8     # epochs sin mejora antes de parar
EARLY_STOPPING_MIN_DELTA = 0.0   # mejora minima de AUC para contar como mejora
```

Salidas en SAVE_DIR: pesos del modelo (best_mito_classifier.pth), curvas de entrenamiento y AUC,
un Excel con las metricas por epoca y un CSV de metadata del mejor epoch.


### clasificacion.py

Clasifica trayectorias con la CNN entrenada. Segun el CSV de entrada:

- Con columna de clase: modo evaluacion. Excluye las aisladas por etiqueta, clasifica el resto con
  la red y calcula metricas (recall, precision, F1, kappa, AUC-ROC y average precision).
- Sin columna de clase: modo inferencia. Detecta las aisladas con la Etapa 1 del EDT (el mismo
  criterio que clasificacion_EDT.py) y clasifica el resto con la red.

Configuracion en el bloque principal (parte final del archivo):

```
MODEL_PATH = "best_mito_classifier.pth"

# Opcion A: una sola imagen
TIF_PATH = "../datos/imagenes_nuevas/SUboligo_02_3_merged.tif"
CSV_PATH = "../datos/imagenes_nuevas/SUboligo_02_3_merged_anotaciones.csv"

# Opcion B: listas manuales
# TIF_PATH = ["datos/img1.tif", "datos/img2.tif"]
# CSV_PATH = ["datos/img1.csv", "datos/img2.csv"]

# Opcion C: directorio completo
# TIF_PATH, CSV_PATH = load_pairs_from_dir("data_augmentation")

SAVE_DIR = "resultados_clasificacion"   # curvas ROC y PR en modo evaluacion
```

La llamada a classify_from_csv incluye los parametros de la Etapa 1 del EDT, que solo se usan en
modo inferencia. Sus valores por defecto son los optimos empiricos y normalmente no hay que
tocarlos:

```
classify_from_csv(
    MODEL_PATH, TIF_PATH, CSV_PATH,
    save_dir=SAVE_DIR,
    umbral_verde=0.0,   # t_g de la mascara de soporte M_g = {I_g > 0}
    umbral_dist=0.0,    # theta: aislada si d_min > theta
    factor_rojo=0.25,   # t_r = factor_rojo * Otsu(rojo>0) para el watershed
)
```

Ejecucion. Para el ensemble se ejecuta varias veces (en GPU si se dispone de ella):

```
CUDA_VISIBLE_DEVICES='0' python3 clasificacion.py
```

Salida: por cada CSV de entrada se escribe <nombre>_clasificado.csv con las columnas clasificacion,
prob_borde, prob_interior y, en modo inferencia, dist_verde_px. Tambien se escribe un log detallado
por trayectoria. En modo evaluacion se guardan las curvas ROC y PR en SAVE_DIR.


### promediar_clasificacion.py

Combina por soft voting los CSVs de varios runs de inferencia: promedia la probabilidad de la clase
Interior por trayectoria y aplica el umbral de decision (optimo por F1, o fijo si se indica). Se
necesitan al menos dos runs.

Para preparar la entrada, se ejecuta clasificacion.py varias veces y se recogen los CSVs
<nombre>_clasificado.csv (con la columna prob_interior) en un mismo directorio.

Configuracion en la parte superior del archivo:

```
DIRECTORIO_ENTRADA = "../resultados/early_stopping2/clasificado"  # CSVs de los runs
DIRECTORIO_SALIDA  = None   # None -> mismo directorio que la entrada
UMBRAL_MANUAL      = None   # None -> se calcula el optimo por F1
```

Los CSVs generados por el propio script (prefijo ensemble_) se excluyen automaticamente de la
entrada, de modo que no se re-ingieren en ejecuciones sucesivas.

Ejecucion (usando el cuadro de configuracion):

```
python promediar_clasificacion.py
```

Los argumentos de consola, si se pasan, tienen prioridad sobre la configuracion del archivo:

```
python promediar_clasificacion.py ../resultados/clasificado
python promediar_clasificacion.py ../resultados/clasificado --salida ../resultados/ensemble
python promediar_clasificacion.py ../resultados/clasificado --umbral 0.4
```

Salidas: ensemble_clasificacion.csv (todas las imagenes juntas), un CSV por imagen en el
subdirectorio por_imagen (con columnas X, Y, Clase, clasificacion, prob_interior), y, cuando hay
etiquetas, ensemble_metricas.csv junto con las curvas ROC y PR del ensemble en SVG.


### imagenes_clasificado.py

Superpone la clasificacion sobre la imagen (PNG y TIFF) y, opcionalmente, calcula metricas por
imagen comparando una columna de ground truth con una de prediccion.

Configuracion en el bloque principal (parte final del archivo):

```
DIRECTORIO_IMAGENES = "../datos/definitivos/test"                              # TIFFs de entrada
DIRECTORIO_CSV      = "../resultados/early_stopping2/clasificado/por_imagen"   # CSVs de entrada
DIRECTORIO_SALIDA   = "../resultados/early_stopping2/clasificado/imagenes"     # PNG, TIFF, metricas

NOMBRE_IMAGEN = "Sub_02_10.tif"
NOMBRE_CSV    = "Sub_02_10_clasificado_test_ensemble.csv"

NOMBRE_BASE   = None            # None -> nombre de la imagen sin extension
SUFIJO_SALIDA = "_clasificada"

col_x = "X"
col_y = "Y"
col_label = "clasificacion"     # columna a representar

calcular_metricas_flag = True   # True para calcular metricas
col_gt   = "Clase"              # columna de ground truth
col_pred = "clasificacion"      # columna de prediccion
```

Ejecucion:

```
python imagenes_clasificado.py
```

Salidas: <base>_clasificada.png, <base>_clasificada.tif y, si se activa el calculo,
<base>_clasificada_metricas.csv.


### representaciones.py

Genera figuras SVG a partir de los runs de entrenamiento (curvas individuales y combinadas). La
carpeta de runs se configura dentro del archivo (RUNS_DIR). El script acepta argumentos opcionales.

Ejecucion:

```
python representaciones.py                  # todas las figuras
python representaciones.py --only-combined  # solo las figuras combinadas
python representaciones.py --run run1       # solo un run concreto
python representaciones.py --run run1 run2  # varios runs concretos
```


## Arquitectura de la CNN

La red MitochondriaContextCNN (definida en modelo_CNN.py) es de tipo inception, con ramas paralelas
de convoluciones 3x3 y 5x5 cuyas caracteristicas se combinan. Usa normalizacion por lotes tras las
convoluciones y dropout en la parte densa. La entrada son parches de dos canales centrados en cada
trayectoria y la salida distingue Interior frente a Borde.

Sobre las curvas de entrenamiento: que la perdida de validacion quede por debajo de la de
entrenamiento es esperable por el dropout y por la diferencia entre los modos de entrenamiento y
evaluacion de la normalizacion por lotes; no indica un problema. El checkpoint se selecciona por el
AUC de validacion en lugar de por la perdida de validacion, para no ajustarse al ruido de la meseta.


## Formato de los CSV de salida

CSV de inferencia de la CNN (<nombre>_clasificado.csv):

- clasificacion: Borde, Interior o Aislada.
- prob_borde: probabilidad de la clase Borde.
- prob_interior: probabilidad de la clase Interior.
- dist_verde_px: distancia al verde (d_min), solo en modo inferencia.

CSV del ensemble por imagen (<id_imagen>_ensemble.csv):

- X, Y: coordenadas.
- Clase: etiqueta real, si estaba disponible.
- clasificacion: prediccion del ensemble.
- prob_interior: probabilidad media de Interior sobre los runs.