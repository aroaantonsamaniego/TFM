import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from modelo_CNN import MitochondriaContextCNN
from funciones_auxiliares import build_dataset_from_csv


def prepare_loaders(tif_path, csv_path, patch_size=64, batch_size=32, val_split=0.2):
    '''
    Construye los DataLoaders (paquetes de datos) de entrenamiento y validacion directamente
    desde el TIFF y el CSV de etiquetas.

    Args:
        tif_path   (str):   Ruta al archivo .tif de 2 canales.
        csv_path   (str):   Ruta al CSV con columnas y, x, clase.
        patch_size (int):   Tamaño del recorte (por defecto 64).
        batch_size (int):   Tamaño del batch (por defecto 32).
        val_split  (float): Fracción del dataset para validacion (por defecto 0.2).

    Returns:
        tuple: (train_loader, val_loader)
    '''
    
    dataset  = build_dataset_from_csv(tif_path, csv_path, patch_size) #cargamos datos
    val_size = int(len(dataset) * val_split) #definimos que procentaje de datos van para validacion
    trn_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [trn_size, val_size]) #que los datos tomados para cada grupo sean aletorios

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) #utilizamos dataloader para envairle a la red los datos de 32 en 32
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False) # en entrenamiento shuffle true para que se cojan cada vez en un orden

    print(f"Train: {trn_size} muestras | Val: {val_size} muestras")
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    '''
    Ejecuta el ciclo completo de entrenamiento y validacion.

    Args:
        model        (nn.Module): MitochondriaContextCNN.
        train_loader (DataLoader): Datos de entrenamiento.
        val_loader   (DataLoader): Datos de validacion.
        num_epochs   (int): Numero de vueltas para entrenamiento.
        lr           (float): Tasa de aprendizaje inicial.

    Returns:
        tuple: (modelo_entrenado, train_losses, val_losses)
    '''
    #Si hay tarjeta grafica la utilizamos
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model     = model.to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) #se encarga de los pesos de la red
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5) #ajusta en caso de detectar aprendizaje lento

    best_val_acc = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train() #comando interno de pytorch que pone la red en modo entrenamiento (hay dropout)
        train_loss, train_correct, train_total = 0, 0, 0

        for patches, labels in train_loader:
            patches = patches.to(device)               
            labels  = labels.squeeze(1).to(device) #para convertir en lista plana(conflicto con la funcion criterio)     

            optimizer.zero_grad() #borrar memoria optimizador
            outputs = model(patches) #le pasamos datos a la red para que nos devuelva predicciones
            loss    = criterion(outputs, labels) #comparamos salida red con la realidad
            loss.backward() #recorre la red al reves para encontrar los fallos
            optimizer.step() #se ajustan los pesos 'malos' detectados con backward

            train_loss    += loss.item()
            _, predicted   = outputs.max(1)
            train_total   += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        
        model.eval() # comando interno pytorch poner red modo validacion
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad(): #para que no guarde el historial y sepa que no hay backward, solo camparamos
            for patches, labels in val_loader:
                patches = patches.to(device)
                labels  = labels.squeeze(1).to(device)

                outputs = model(patches)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() #error total del bloque
                _, predicted = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total #precicion de acierto entrenamiento
        val_acc   = 100. * val_correct   / val_total #precision de acierto validacion

        train_losses.append(train_loss / len(train_loader)) #error medio de la vuelta
        val_losses.append(val_loss     / len(val_loader))
        scheduler.step(val_losses[-1]) 

        print(f'Epoch {epoch+1:2d}: Train {train_acc:.1f}% | Val {val_acc:.1f}% | '
              f'Train Loss {train_losses[-1]:.3f} | Val Loss {val_losses[-1]:.3f}')

        #Guardado del modelo, compara con mejor valor de validacion registrado en cualquier momento
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_mito_classifier.pth') #guarda todos los datos de la red actualizados (son los que se utilizan para una clasificacion fuera del training)
            print(f'  → Modelo guardado (val_acc={val_acc:.1f}%)')

    return model, train_losses, val_losses


#Ejecucion del codigo
if __name__ == "__main__":
    TIF_PATH = "imagen.tif"   # <-- ruta a tu archivo TIFF
    CSV_PATH = "etiquetas.csv"  # <-- ruta a tu CSV con columnas y, x, clase

    train_loader, val_loader = prepare_loaders(TIF_PATH, CSV_PATH)

    model = MitochondriaContextCNN(num_channels=2, num_classes=3)
    train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001)
