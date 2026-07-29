# Clasificación de trayectorias estáticas en microscopía de fluorescencia de dos canales

Este repositorio implementa la clasificación de trayectorias estáticas en imágenes TIFF de
microscopía de fluorescencia de dos canales (canal rojo: trayectorias estáticas; canal verde:
taryectorias elípticas y no homogéneas). Cada trayectoria estática se asigna a una de las tres categorias:

- Interior
- Borde
- Aislada

El proyecto contiene dos métodos de clasificación:

1. Un clasificador clásico basado en la transformada de distancia euclidea (EDT), en dos etapas.
2. Un clasificador basado en Deep Learning (CNN) con postprocesado por promediado de probabilidades.

El clasificador EDT cumple un doble papel: es un clasificador clásico independiente y, además, actua
como pre-filtro de aisladas antes de la CNN.


## Organización de módulos

El código está organizado de forma que los dos modelos de clasificación sean independientes, con un único módulo
común de utilidades:

- funciones_auxiliares.py es el módulo común. Contiene la carga de imágenes y CSV, la extracción de
  patches, el Dataset de entrenamiento, la detección de aisladas de la Etapa 1
  (detectar_aisladas_EDT) y la construcción de territorios watershed (construir_territorios). El
  import de PyTorch es opcional: solo se necesita para el Dataset de entrenamiento, de modo que el
  método clasico puede usar este módulo sin PyTorch instalado.
- clasificacion_EDT.py y run_EDT.py conforman el clasificador clásico.
- clasificacion.py, entrenamiento.py, modelo_CNN.py conforman el clasificador por medio de la CNN.


## Listado de archivos

- funciones_auxiliares.py    Módulo común: carga de datos, patches, Dataset, detección de aisladas.
- modelo_CNN.py              Arquitectura de la CNN (MitochondriaContextCNN).
- entrenamiento.py           Entrenamiento de la CNN.
- clasificacion.py           Inferencia con la CNN entrenada.
- clasificacion_EDT.py       Clasificador clasico EDT en dos etapas.
- run_EDT.py                 Lanzador de clasificacion_EDT.py al estilo "configurar y descomentar".
- promediar_clasificacion.py Ensemble por soft voting sobre varios runs.
- imagenes_clasificado.py    Visualización de resultados sobre la imagen y cálculo de métricas por imagen.
- representaciones.py        Figuras SVG a partir de los runs de entrenamiento.
- data_augmentation.py       Aumento de datos (imagenes y CSV) con albumentations.
- anotador_particulas.py     Herramienta interactiva de anotación manual.


## Requisitos

- Python 3.8 o superior.
- PyTorch (con CUDA si se dispone de GPU). Solo necesario para el modelo de la CNN.
- numpy, scipy, scikit-image, scikit-learn, pandas, matplotlib, tifffile, opencv-python.
- albumentations (solo para data_augmentation.py).

Instalación de dependencias (ejemplo con CUDA 11.8):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-image scikit-learn pandas matplotlib tifffile opencv-python albumentations openpyxl
```

Para ejecutar únicamente el clasificador clásico (clasificacion_EDT.py) no hace falta PyTorch:

```
pip install numpy scipy scikit-image scikit-learn pandas matplotlib tifffile
```


## Formato de los datos

Imagen TIFF de entrada:

- Dos canales. Canal 0: rojo (trayectorias estaticas). Canal 1: verde (trazas).

CSV de anotaciones de entrada. Columnas mínimas:

- X, Y: coordenadas del centroide de cada trayectoria.
- clase (opcional): interior, borde o aislada.

El comportamiento depende de si el CSV incluye la columna de clase:

- Con columna de clase: Las aisladas se excluyen por etiqueta y el resto se
  compara contra las etiquetas reales para calcular métricas.
- Sin columna de clase: Las aisladas se detectan con el filtro diseñado para aisladas.


## Flujo de trabajo completo

```
1. Anotación
   python anotador_particulas.py
   Genera un CSV de anotaciones por imagen.

2. Aumento de datos (opcional)
   python data_augmentation.py
   Genera imagenes y CSV aumentados.

3. Clasificador clásico
   python run_EDT.py
  

4. Entrenamiento de la CNN (repetir varias veces para deep ensemble)
   CUDA_VISIBLE_DEVICES=0 python3 entrenamiento.py
   Genera best_mito_classifier.pth, curvas y logs.
   Genera un CSV <nombre>_clasificado.csv por imagen test.

6. Ensemble
   python promediar_clasificacion.py
   Promedia las probabilidades de los distintos runs.

7. Visualización y métricas por imagen
   python imagenes_clasificado.py
   Genera PNG, TIFF (4 canales) y CSV de métricas.
```


## Ejecución detallada de cada script
### anotador_particulas.py

Herramienta interactiva para anotar manualmente las trayectorias. Detección automática de puntos
sobre el canal rojo, sliders de contraste, y guardado en CSV.

Configuración en la parte superior del archivo (la selección del archivo puede hacerse directamente con el cuadro de diálogo que aparece al ejecutar)

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

Ejecución:

```
python anotador_particulas.py
```

Salida: un CSV por imagen con las columnas X, Y y la clase asignada.


### data_augmentation.py

Genera versiones aumentadas de cada par imagen más CSV (rotaciones, flips y demás transformaciones
definidas en el diccionario TRANSFORMS), transformando de forma coherente las coordenadas de las
trayectorias.

Configuracion en la parte superior del archivo:

```
INPUT_DIR  = "../datos/definitivos/training"           # imagenes y CSV originales
OUTPUT_DIR = "../datos/definitivos/data_augmentation"  # salida aumentada
IMAGE_EXT  = ".tif"
```

Ejecución:

```
python data_augmentation.py
```

Salida: para cada imagen de entrada, un TIFF y un CSV por transformacion, con nombre
<base>_<transformacion>.tif y <base>_<transformacion>.csv.


### clasificacion_EDT.py y run_EDT.py

Clasificador clásico en dos etapas:

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

Rutas y parámetros:

```
# Modos de un par
TIF = "../datos/imagenes_nuevas/test/SUboligo_02_3_merged.tif"
CSV = "../datos/imagenes_nuevas/test/SUboligo_02_3_merged_anotaciones.csv"

# Modos de directorio
DIR = "../datos/definitivos"

# Etapa 1 (aisladas)
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

Ejecución:

```
python run_EDT.py
```

### entrenamiento.py

Entrena la CNN. Excluye automáticamente las trayectorias etiquetadas como aisladas y divide el
conjunto en entrenamiento y validaciÓn. Incluye early stopping opcional sobre el AUC de validación,
reducción del learning rate si no mejora en x épocas y guardado del mejor modelo por AUC de validación.

Ejecución en GPU. Se indica el identificador de la GPU con CUDA_VISIBLE_DEVICES:

```
CUDA_VISIBLE_DEVICES=0 python3 entrenamiento.py
```

Configuración en el bloque principal (parte final del archivo). Hay tres formas de indicar los
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
un Excel con las métricas por epoca, un CSV de metadata del mejor epoch y los CSV de clasificación de cada uno de los archivos de test.


### clasificacion.py

Clasifica trayectorias con la CNN entrenada. Según el CSV de entrada:

- Con columna de clase: Excluye las aisladas por etiqueta, clasifica el resto con
  la red y calcula métricas (recall, precision, F1, kappa, AUC-ROC y average precision).
- Sin columna de clase: Detecta las aisladas con el filtro diseñado para las aisladas por EDT y clasifica el resto con la red.

Configuración en el bloque principal (parte final del archivo):

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

Ejecución:

```
CUDA_VISIBLE_DEVICES=0 python3 clasificacion.py
```

Salida: por cada CSV de entrada se escribe <nombre>_clasificado.csv con las columnas clasificacion,
prob_borde, prob_interior. También se escribe un log detallado
por trayectoria.


### promediar_clasificacion.py

Combina por soft voting los CSVs de varios runs de inferencia: promedia la probabilidad de la clase
Interior por trayectoria y aplica el umbral de decisión (óptimo por F1, o fijo si se indica). 

Configuración en la parte superior del archivo:

```
DIRECTORIO_ENTRADA = "../resultados/early_stopping2/clasificado"  # CSVs de los runs
DIRECTORIO_SALIDA  = None   # None -> mismo directorio que la entrada
UMBRAL_MANUAL      = None   # None -> se calcula el optimo por F1
```

Ejecución:

```
python promediar_clasificacion.py
```

Los argumentos de consola, si se pasan, tienen prioridad sobre la configuración del archivo:

```
python promediar_clasificacion.py ../resultados/clasificado
python promediar_clasificacion.py ../resultados/clasificado --salida ../resultados/ensemble
python promediar_clasificacion.py ../resultados/clasificado --umbral 0.4
```

Salidas: ensemble_clasificacion.csv (todas las imagenes juntas), un CSV por imagen en el
subdirectorio por_imagen (con columnas X, Y, Clase, clasificacion, prob_interior), y, cuando hay
etiquetas, ensemble_metricas.csv junto con las curvas ROC y PR del ensemble en SVG.


### imagenes_clasificado.py

Superpone la clasificación sobre la imagen (PNG y TIFF) y, opcionalmente, calcula métricas por
imagen comparando una columna de ground truth con una de prediccion.

Configuración en el bloque principal (parte final del archivo):

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

Ejecución:

```
python imagenes_clasificado.py
```

Salidas: <base>_clasificada.png, <base>_clasificada.tif y, si se activa el cálculo,
<base>_clasificada_metricas.csv.


### representaciones.py

Genera figuras SVG a partir de los runs de entrenamiento (curvas individuales y combinadas). La
carpeta de runs se configura dentro del archivo (RUNS_DIR). El script acepta argumentos opcionales.

Ejecución:

```
python representaciones.py                  # todas las figuras
python representaciones.py --only-combined  # solo las figuras combinadas
python representaciones.py --run run1       # solo un run concreto
python representaciones.py --run run1 run2  # varios runs concretos
```


## modelo_CNN.py

Se definen dos arquitecturas de CNN, una tipo lineal y otra tipo inception para el estudio de ambas.

Este archivo nunca se ejecuta solo debe estar disponible en el mismo directorio que entrenamiento.py.

