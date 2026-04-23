'''
Este codigo se encarga de establecer la estructura de la CNN para clasificacion de la posicion de particulas
en funcion de si se encuentran en el interior, exterior (no deberia) o sobre el borde.
La red analizara particula por particula centrandose en la poscion para esta y comparara loscalmente
respecto de las trayectorias que se observan a su alrededor para determinar la posicion de esta.
'''

import torch.nn as nn # contiene bloques de construccion de la red(capas convolucionales, funciones de activacion...)
import torch.nn.functional as F #Para utilizar ReLU


class MitochondriaContextCNN(nn.Module): #definimos la red como un modelo entrenable de PyTorch con nn.Module
    
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
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(16) #normaliza para evitar valores extremos
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2) #reduce a la mitad el tamanyo de un canal
        self.dropout = nn.Dropout(0.25) #'apaga' aleatoriamente el 25% de los datos para evitar overfitting
        
        # Global Average Pooling + Clasificador final
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) #hace la media de un canal para obtener un unico valor
        self.fc = nn.Linear(128, num_classes) #hace la operacion matematica que me devuelve un vector con 2 valores
        # que indican la probabilidad de pertenecer a cada categoria: borde, no borde
        
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
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x) #para reducir overfitting, solo se utiliza en el training
        
        # Global Average Pooling
        x = self.global_avg_pool(x) 
        x = x.view(x.size(0), -1) #aplana los datos en un unico vector para hacer la clasificacion
        # el primer argumento me indica el numero de imagenes totales que le hemos pasado a la red ya que no 
        #solemos enviarle una por una sino que le enviaremos un grupo. 
        
        # Clasificacion final
        x = self.fc(x)
        return x
