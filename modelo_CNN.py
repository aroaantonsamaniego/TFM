'''
Este codigo se encarga de establecer la estructura de la CNN para clasificacion de la posicion de particulas
en funcion de si se encuentran en el interior, exterior (no deberia) o sobre el borde.
La red analizara particula por particula centrandose en la poscion para esta y comparara loscalmente
respecto de las trayectorias que se observan a su alrededor para determinar la posicion de esta.
'''
import torch
import torch.nn as nn # contiene bloques de construccion de la red(capas convolucionales, funciones de activacion...)
import torch.nn.functional as F #Para utilizar ReLU


class MitochondriaContextCNN_1(nn.Module): #definimos la red como un modelo entrenable de PyTorch con nn.Module
    
    def __init__(self, num_channels=2, num_classes=2):
        '''
        Define la arquitectura de la red: 4 bloques convolucionales con 
        normalizacion de batch y un clasificador basado en Global Average Pooling.
        
        Args:
            num_channels (int): Numero de mapas de entrada (en principio 2: estaticas y elipticas)
            num_classes (int): Numero de categorias de salida (Borde, No borde).
        '''
        super().__init__() 
        
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1) #las 2 primeras entradas define el numero de capas de entrada y salida
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) #el tamaño del kernel es el tamaño del filtro, en este caso 3x3 pixeles
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #el padding=1 introduce un borde de 0 alrededor para visulizar toda la info
        #self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(16) #normaliza para evitar valores extremos
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        #self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2) #reduce a la mitad el tamanyo de un canal
        self.dropout = nn.Dropout(0.35) #'apaga' aleatoriamente el 25% de los datos para evitar overfitting
        
        # Global Average Pooling + Clasificador final
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) #hace la media de un canal para obtener un unico valor
        self.fc1 = nn.Linear(64, 32) #capa densa intermedia: combina los patrones espaciales antes de clasificar
        self.fc2 = nn.Linear(32, num_classes) #hace la operacion matematica que me devuelve un vector con 2 valores
        # que indican la probabilidad de pertenecer a cada categoria: borde, interior
        
    def forward(self, x): 
        '''
        Se define el funcionamiento de la red.
        Bloques 1-3:
        1. Se aplica la convolucion
        2. Se normaliza para evitar valores muy extremos que puedan desestabilizar la red
        3. Se le pasan los datos a la funcion de activacion. Hemos escogido ReLU por ser la mas comun pero 
           se podria cambiar.
        4. Se hace un pooling para reducir las dimensiones de cada canal.

        Bloque 4:
        Hace lo mismo que los bloques anteriores pero al final en lugar de hacer un pooling hace un dropout 
        para evitar sobreentrenamiento. Su valor se puede ajustar si vemos que hay overfitting (subirlo).
        En algunos casos también funciona bien ponerlo despues del average pooling final.
        '''
        # Bloque 1
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) 
        # Bloque 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Bloque 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Bloque 4
        #x = F.relu(self.bn3(self.conv3(x)))
        #x = self.dropout(x) #para reducir overfitting, solo se utiliza en el training
        
        # Global Average Pooling
        x = self.global_avg_pool(x) 
        x = x.view(x.size(0), -1) #aplana los datos en un unico vector para hacer la clasificacion
        # el primer argumento me indica el numero de imagenes totales que le hemos pasado a la red ya que no 
        #solemos enviarle una por una sino que le enviaremos un grupo. 
        
        # Clasificacion final: capa densa intermedia + dropout + clasificador
        x = self.dropout(F.relu(self.fc1(x))) #ReLU + dropout para regularizar la capa densa
        x = self.fc2(x)
        return x


'''
Arquitectura CNN de 2 ramas paralelas para clasificación de partículas
en borde / no borde de orgánulos mitocondriales.

Cada rama replica la estructura de la red lineal que mejores resultados
dio, especializándose en una escala espacial distinta:

  Rama A (3×3): detecta patrones locales progresivamente — bordes,
    texturas y gradientes en el entorno inmediato de la partícula.

  Rama B (5×5): detecta contexto más amplio desde el primer bloque —
    presencia y distribución de estructura verde en una zona mayor
    alrededor de la partícula, clave para distinguir borde de no borde.

Ambas ramas tienen la misma profundidad (3 bloques con pool en todos)
y producen el mismo número de mapas (64), contribuyendo por igual
a la decisión final tras la concatenación.

El Dropout se aplica entre las dos capas densas del clasificador,
que es donde tiene sentido lógico: obliga a fc2 a no depender de
ninguna neurona concreta de fc1, distribuyendo el aprendizaje de la
decisión final. No se aplica antes de fc1 para no descartar
aleatoriamente filtros convolucionales completos aprendidos por las ramas.

Flujo de dimensiones (parche 64×64):
    Entrada  → (N,   2, 64, 64)
    ┌────────────────────────────┐
    Rama A                    Rama B
    3×3 + pool                5×5 + pool
    (N, 16, 32, 32)           (N, 16, 32, 32)
    3×3 + pool                5×5 + pool
    (N, 32, 16, 16)           (N, 32, 16, 16)
    3×3 + pool                5×5 + pool
    (N, 64,  8,  8)           (N, 64,  8,  8)
    └──────────┬─────────────────┘
           cat → (N, 128,  8,  8)
           GAP → (N, 128)
           fc1 + ReLU → (N, 64)
           Dropout(0.35)
           fc2 → (N, 2)
'''




class MitochondriaContextCNN(nn.Module):

    def __init__(self, num_channels=2, num_classes=2):
        '''
        Args:
            num_channels (int): Canales de entrada (default: 2).
            num_classes  (int): Clases de salida (default: 2: Borde / No borde).
        '''
        super().__init__()

        # ── Rama A: 3 bloques con kernel 3×3 ─────────────────────────────────
        # Misma estructura que la red lineal que mejor funcionó.
        # Los 3 MaxPool reducen 64×64 → 32×32 → 16×16 → 8×8.
        self.ra_conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, padding=1)
        self.ra_bn1   = nn.BatchNorm2d(8)

        self.ra_conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.ra_bn2   = nn.BatchNorm2d(16)

        self.ra_conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.ra_bn3   = nn.BatchNorm2d(32)

        # ── Rama B: 3 bloques con kernel 5×5 ─────────────────────────────────
        # padding=2 en el 5×5 mantiene el mismo tamaño espacial que el 3×3
        # con padding=1 → ambas ramas producen mapas (N, 64, 8, 8) y se
        # pueden concatenar directamente sin redimensionado.
        self.rb_conv1 = nn.Conv2d(num_channels, 8, kernel_size=5, padding=2)
        self.rb_bn1   = nn.BatchNorm2d(8)

        self.rb_conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.rb_bn2   = nn.BatchNorm2d(16)

        #self.rb_conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        #self.rb_bn3   = nn.BatchNorm2d(32)

        # Pool compartido entre ambas ramas (mismo stride y kernel)
        self.pool = nn.MaxPool2d(2, 2)

        # ── Global Average Pooling ────────────────────────────────────────────
        # (N, 128, 8, 8) → (N, 128)
        # 128 = 64 (rama A) + 64 (rama B)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Clasificador con capa densa intermedia ────────────────────────────
        # fc1: combina los 128 patrones (64 de cada rama) en 64 neuronas.
        # ReLU: introduce no linealidad en el clasificador.
        # Dropout(0.35): aplicado ENTRE fc1 y fc2 — obliga a fc2 a no
        #   depender de ninguna neurona concreta de fc1, distribuyendo
        #   el aprendizaje de la decisión final. No se pone antes de fc1
        #   para no descartar aleatoriamente filtros de las ramas.
        # fc2: proyección final 64 → 2 (logits Borde / No borde).
        self.fc1     = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(16, num_classes)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): Batch de parches (N, 2, 64, 64).

        Returns:
            torch.Tensor: Logits (N, 2). Aplicar softmax para probabilidades.
        '''
        # ── Rama A: 3×3 ───────────────────────────────────────────────────────
        ra = self.pool(F.relu(self.ra_bn1(self.ra_conv1(x))))  # (N, 16, 32, 32)
        ra = self.pool(F.relu(self.ra_bn2(self.ra_conv2(ra)))) # (N, 32, 16, 16)
        #ra = self.pool(F.relu(self.ra_bn3(self.ra_conv3(ra)))) # (N, 64,  8,  8)

        # ── Rama B: 5×5 ───────────────────────────────────────────────────────
        rb = self.pool(F.relu(self.rb_bn1(self.rb_conv1(x))))  # (N, 16, 32, 32)
        rb = self.pool(F.relu(self.rb_bn2(self.rb_conv2(rb)))) # (N, 32, 16, 16)
        #rb = self.pool(F.relu(self.rb_bn3(self.rb_conv3(rb)))) # (N, 64,  8,  8)

        # ── Concatenación ─────────────────────────────────────────────────────
        x = torch.cat([ra, rb], dim=1)                         # (N, 128, 8, 8)

        # ── GAP + aplanado ────────────────────────────────────────────────────
        x = self.global_avg_pool(x)                            # (N, 128, 1, 1)
        x = x.view(x.size(0), -1)                             # (N, 128)

        # ── Clasificador: fc1 → ReLU → Dropout → fc2 ─────────────────────────
        x = self.dropout(F.relu(self.fc1(x)))                  # (N, 64)
        x = self.fc2(x)                                        # (N,  2)
        return x