import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from CNN_classe import Classifier
import pandas as pd
import numpy as np
import torch.nn as nn
from time import time


device = "cuda"

def collate_fn(batch):
    """
        fonction qui permet de mttre toutes les données sur GPU pour éviter des échange GPU-CPU à chaque batch et détruire mon CPU
    """
    pixels, labels = zip(*batch)
    pixels = torch.stack(pixels).to(device, non_blocking=True)
    labels = torch.tensor(labels).to(device, non_blocking=True)
    return pixels, labels

class CSVDataset(Dataset):
    """
        classe pour faciliter la mise en forme des données 
    """
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=1)
        self.labels = self.data.iloc[:, 0]
        self.pixels = self.data.iloc[:, 1:].values.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = torch.tensor(self.labels[index])
        pixels = torch.tensor(self.pixels[index]).reshape((1, 28, 28))
        return pixels, label

def trainning_all_models(list_models: list[Classifier], nb_generation: int) -> list[Classifier]:
    """
        fonction qui permet d'entraîner tous les modèles de la liste
    """
    # on regarde si on peut utiliser le GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Utilisation du GPU pour l'entraînement")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("Utilisation du CPU pour l'entraînement")

    num_epochs = 12    # nombre de fois qu'on va parcourir le data_set d'entraienemnt lors de ce dernier
    batch_size = 128   # par combien les images vont passer en même temps dans le réseau pour l'entraienement 


    # chargement des données
    train_dataset = CSVDataset('dataset/sign_mnist_train/sign_mnist_train.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = CSVDataset('dataset/sign_mnist_test/sign_mnist_test.csv')
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, collate_fn=collate_fn)

    model_num = 0
    for model in list_models:
        #si le modèle provient d'un génération d'avant, pas besoin de le réentrainer
        if model.accuracy > 0:
            continue
        
        model.to(device)
        criterion = nn.CrossEntropyLoss()                     # initiatlisation de la fonction de loss
        optimizer = optim.Adam(model.parameters(), lr=0.0001) # init de l'optimiseur
        start = time()

        try:
            for epoch in range(num_epochs):
                running_loss = 0.0
                model.train()
                for i, data in enumerate(train_loader):
                    pixels, labels = data
                    optimizer.zero_grad()
                    outputs = model(pixels)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                print(f'Époque {epoch + 1}, perte : {running_loss / len(train_loader)}')
        except Exception as e:
            print(f"Erreur pendant l'entraînement du modèle: {e}")
            model.accuracy = -1
            continue

        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for data in test_loader:
                pixels, labels = data
                outputs = model(pixels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        model.accuracy = correct / total
        end = time()
        print(f"Accuracy du modèle : {model.accuracy} en {end - start} sec\ngeneration: {nb_generation}, model numero: {model_num}, nb_para: {model.count_parameters()}")
        model_num += 1
        model.time_training = end - start

    return list_models
