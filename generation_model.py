# Description: Ce fichier contient les fonctions qui vont permettre de générer les modèles pour notre population

from CNN_classe import Classifier
import random

def create_new_models(liste_params : list[list[int]]) -> list[Classifier]:
    """
        Cette fonction va permettre de créer les différents modèles pour notre population
        liste_params: liste de paramètres pour chaque modèle
    """
    list_models = [ Classifier(para) for para in liste_params]

    return list_models

def gene_liste_models(nb_models : int, nb_couhes : int) -> list[Classifier]:
    """
        Fonction qui va permettre de générer une liste de modèles
        nb_models: nombre de modèles que l'on veut générer
        nb_couhes: nombre de couches que l'on veut pour chaque modèle
        les paramètres autres que le nombre de couches sont tous chosis aléatoirement dans les bornes que l'on a fixé
    """
    liste_models = []
    for i in range(nb_models):
        para_model = []
        borne_max = [124,250,500,500]
        for j in range(nb_couhes):
            para_couche = [random.randint(25, borne_max[j]), random.randint(2, 5), random.uniform(0, 0.6)]
            para_model.append(para_couche)

        new_model = Classifier(para_model)
        print(new_model)
        liste_models.append(new_model)
        
    return liste_models