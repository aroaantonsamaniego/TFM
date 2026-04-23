import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc 

from modelo_CNN import MitochondriaContextCNN
from funciones_auxiliares import build_dataset_from_csv


def prepare_loaders(tif_path, csv_path, patch_size=64, batch_size=32, val_split=0.2):
    '''
    Construye los DataLoaders (paquetes de datos) de entrenamiento y validacion directamente
    desde el TIFF y el CSV de etiquetas.

    Args:
        tif_path   (str):   Ruta al archivo .tif de 2 canales.
        csv_path   (str):   Ruta al CSV con columnas y, x, clase.
        patch_size (int):   Tamanyo del recorte (por defecto 64).
        batch_size (int):   Tamanyo del batch (por defecto 32).
        val_split  (float): Fraccion del dataset para validacion (por defecto 0.2).

    Returns:
        tuple: (train_loader, val_loader)
    '''
    dataset  = build_dataset_from_csv(tif_path, csv_path, patch_size) #cargamos datos
    val_size = int(len(dataset) * val_split) #definimos que porcentaje de datos van para validacion
    trn_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [trn_size, val_size]) #que los datos tomados para cada grupo sean aleatorios

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  #utilizamos dataloader para enviarle a la red los datos de 32 en 32
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False) #en entrenamiento shuffle true para que se cojan cada vez en un orden

    print(f"Train: {trn_size} muestras | Val: {val_size} muestras")
    return train_loader, val_loader


def plot_roc_curve(val_scores, val_labels, save_path=None):
    '''
    Calcula y representa la curva ROC sobre los datos de validacion del mejor modelo.

    La curva ROC muestra la relacion entre la tasa de verdaderos positivos (TPR)
    y la tasa de falsos positivos (FPR) para todos los umbrales de decision posibles.
    El area bajo la curva (AUC) resume la calidad global del clasificador:
        AUC = 1.0  ->  clasificador perfecto
        AUC = 0.5  ->  clasificador aleatorio (sin capacidad discriminativa)

    La puntuacion usada es la probabilidad de la clase Borde (indice 1) producida
    por softmax, que es continua y permite trazar la curva completa.

    Args:
        val_scores (list of float): Probabilidades de clase Borde para cada muestra.
        val_labels (list of int):   Etiquetas reales (0=No borde, 1=Borde).
        save_path  (str | None):    Ruta opcional para guardar la figura en disco.

    Returns:
        float: Valor del AUC.
    '''
    fpr, tpr, _ = roc_curve(val_labels, val_scores)
    roc_auc     = auc(fpr, tpr)

    print(f"AUC-ROC (validacion): {roc_auc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'Curva ROC  (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.2,
            linestyle='--', label='Clasificador aleatorio (AUC = 0.5)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    ax.set_title('Curva ROC - Validacion del mejor modelo\nClasificador Borde / No borde', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Curva ROC guardada en: {save_path}")

    plt.show()
    return roc_auc


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, roc_save_path=None):
    '''
    Ejecuta el ciclo completo de entrenamiento y validacion.
    Al terminar todas las epochs, genera la curva ROC del mejor modelo
    sobre los datos de validacion.

    Args:
        model         (nn.Module):  MitochondriaContextCNN.
        train_loader  (DataLoader): Datos de entrenamiento.
        val_loader    (DataLoader): Datos de validacion.
        num_epochs    (int):        Numero de vueltas para entrenamiento.
        lr            (float):      Tasa de aprendizaje inicial.
        roc_save_path (str|None):   Ruta opcional para guardar la figura ROC en disco.

    Returns:
        tuple: (modelo_entrenado, train_losses, val_losses)
    '''
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) #se encarga de los pesos de la red
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5) #ajusta en caso de detectar aprendizaje lento

    best_val_auc = 0.0
    train_losses, val_losses = [], []

    # Variables para guardar scores y etiquetas de validacion del mejor epoch
    best_val_scores = None
    best_val_labels = None

    for epoch in range(num_epochs):
        model.train() #comando interno de pytorch que pone la red en modo entrenamiento (hay dropout)
        train_loss = 0

        for patches, labels in train_loader:
            patches = patches.to(device)
            labels  = labels.squeeze(1).to(device) #para convertir en lista plana (conflicto con la funcion criterio)

            optimizer.zero_grad()                #borrar memoria optimizador
            outputs = model(patches)             #le pasamos datos a la red para que nos devuelva predicciones
            loss    = criterion(outputs, labels) #comparamos salida red con la realidad
            loss.backward()                      #recorre la red al reves para encontrar los fallos
            optimizer.step()                     #se ajustan los pesos 'malos' detectados con backward

            train_loss += loss.item()

        model.eval() #comando interno pytorch poner red modo validacion
        val_loss      = 0
        val_scores_ep = []  #probabilidad de clase Borde para cada muestra de validacion
        val_labels_ep = []  #etiqueta real de cada muestra de validacion

        with torch.no_grad(): #para que no guarde el historial y sepa que no hay backward, solo comparamos
            for patches, labels in val_loader:
                patches = patches.to(device)
                labels  = labels.squeeze(1).to(device)

                outputs  = model(patches)
                loss     = criterion(outputs, labels)
                val_loss += loss.item()

                # Probabilidad de la clase positiva (Borde, indice 1) via softmax
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                val_scores_ep.extend(probs)
                val_labels_ep.extend(labels.cpu().numpy())

        train_losses.append(train_loss / len(train_loader)) #error medio de la vuelta
        val_losses.append(val_loss     / len(val_loader))
        scheduler.step(val_losses[-1]) #si el error medio no baja, reduce el learning rate

        # Calcular AUC de validacion de esta epoch
        fpr, tpr, _ = roc_curve(val_labels_ep, val_scores_ep)
        val_auc      = auc(fpr, tpr)

        print(f'Epoch {epoch+1:2d}: Val AUC {val_auc:.4f} | '
              f'Train Loss {train_losses[-1]:.3f} | Val Loss {val_losses[-1]:.3f}')

        # Guardar el modelo cuando mejora el AUC de validacion
        if val_auc > best_val_auc:
            best_val_auc    = val_auc
            best_val_scores = val_scores_ep
            best_val_labels = val_labels_ep
            torch.save(model.state_dict(), 'best_mito_classifier.pth') #guarda todos los datos de la red actualizados
            print(f'  -> Modelo guardado (val_auc={val_auc:.4f})')

    # Curva ROC final sobre los datos de validacion del mejor epoch
    print(f'\nEntrenamiento completado. Mejor AUC de validacion: {best_val_auc:.4f}')
    plot_roc_curve(best_val_scores, best_val_labels, save_path=roc_save_path)

    return model, train_losses, val_losses


#Ejecucion del codigo
if __name__ == "__main__":
    TIF_PATH      = "2_canales_red_std_green_rest.tif"      # <-- ruta a tu archivo TIFF
    CSV_PATH      = "datos_particulas_training.csv"   # <-- ruta a tu CSV con columnas y, x, clase
    ROC_SAVE_PATH = "roc_curve.png"   # <-- dejar None para no guardar en disco

    train_loader, val_loader = prepare_loaders(TIF_PATH, CSV_PATH)

    model = MitochondriaContextCNN(num_channels=2, num_classes=2)
    train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001,
                roc_save_path=ROC_SAVE_PATH)
