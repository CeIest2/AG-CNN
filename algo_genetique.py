from CNN_classe import Classifier
from training import trainning_all_models
from gestion_para_simulation import get_mutation_rate, get_mutation_diff
import random
from generation_model import gene_liste_models
from save_results import save_results
import numpy as np


def gene_coeff_learning(liste_models: list[Classifier]) -> None:
    """
        pour faire de l'optimisation de paramètre au va faire en sorte que plus un modèle est au dessus des autres en termes d'accuracie,
        plus les mutations sur ce derniers seront fines
    
    """
    avg_accuracy = sum(model.accuracy for model in liste_models) / len(liste_models)
    std_accuracy = np.std([model.accuracy for model in liste_models])

    for model in liste_models:
        z_score = (model.accuracy - avg_accuracy) / std_accuracy
        learning_coeff = 1 / (1 + np.exp(-z_score))
        model.learning_coeff = learning_coeff


def selection(list_models : list[Classifier], nb_parents : int) -> list[Classifier]:
    """
        Fonction qui va permettre de sélectionner les meilleurs modèles pour la génération suivante
        list_models: liste des modèles de la génération actuelle
        nb_parents: nombre de modèles que l'on veut garder pour la génération suivante
    """


    # On va trier les modèles en fonction de leur accuracy par ordres décroissant
    list_models.sort(key = lambda x: x.accuracy, reverse = True)

    # On va garder les nb_parents premiers modèles
    parents = [list_models[i] for i in range(nb_parents)]
    return parents




def mutation(liste_models : list[Classifier], nb_muta) -> list[Classifier]:
    """
        Fonction qui va permettre de muter les modèles
        On a trois caractéristiques à faire muter :
            - le nombre de canneaux de sortie (nb filtre)
            - la tailles du noyau
            - le taux de dropout
    """
    t_muta_filtre, t_muta_noyau, t_muta_dropout = get_mutation_rate() # taux de chance de muter
    t_diff_filtre, t_diff_noyau, t_diff_dropout = get_mutation_diff() # taux de différence entre les valeurs
    id_mutated_models = [random.randint(0,len(liste_models))-1 for _ in range( nb_muta)]
    liste_models_mutants = []


    for id in id_mutated_models:
        model = liste_models[id]
        new_para = [[para for para in couche] for couche in model.para]
        for couche in range(len(model.para)):

            # On va muter le nombre de canneaux de sortie
            if random.random() < t_muta_filtre:
                new_para[couche][0] =  int(model.para[couche][0] + (random.uniform(-t_diff_filtre, t_diff_filtre) * model.learning_coef ** 2))

            # On va muter la taille du noyau
            if random.random() < t_muta_noyau:
                new_para[couche][1] += int(random.randint(-t_diff_noyau, t_diff_noyau) * model.learning_coef ** 2)

            # On va muter le taux de dropout
            if random.random() < t_muta_dropout:
                new_para[couche][2] += random.uniform(-t_diff_dropout, t_diff_dropout) * model.learning_coef ** 2

        for para in new_para:

            #on réajuste les paramètres pour qu'ils soient dans les bornes
            #########################
            para[0] = max(1, para[0])
            if para[0] > 256:
                para[0] = 512
            para[1] = max(2, para[1])
            if para[1] > 7:
                para[1] = 7
            para[2] = max(0.0, para[2])
            if para[2] > 0.5:
                para[2] = 0.5


        new_model = Classifier(new_para)
        liste_models_mutants.append(new_model)
    

    return liste_models_mutants

def crossover(liste_models : list[Classifier], nb_crois : int)-> list[Classifier]:
    """
        pour un cross over entre 2 modèles on va simplement faire la moyenne des para des 2 modèles
    """
    id_crois_models = [random.randint(0,len(liste_models)-1) for _ in range(nb_crois * 2)] # je sais que on peut avoir des croissement avec le même éléments mais flemme de gérer ça pour l'instant
    new_generation = []
    for i in range(len(id_crois_models)):
        enfant_para = []
        parent_1, parent_2 = liste_models[id_crois_models[i]].para, liste_models[id_crois_models[-((i + 1)%(len(id_crois_models)-1))]].para

        for couche in range(len(parent_1)):
            # ici on va faire la moyen de toutes les caractéristiques de la couche
            enfant_para.append([int((parent_1[couche][0] + parent_2[couche][0])/2), int((parent_1[couche][1] + parent_2[couche][1])/2), (parent_1[couche][2] + parent_2[couche][2])/2])

        enfant = Classifier(enfant_para)
        new_generation.append(enfant)
    return new_generation



    



def go_through_generation(liste_models : list[Classifier], nb_select : int, nb_crois : int, nb_mut : int, nb_gene :int, size_generation :int) -> list[Classifier]:
    """
        dans cette fonction on va faire passer une génération 
        liste_models: liste des modèles de la génération actuelle
        nb_select:    nombre de modèles que l'on veut garder pour la génération suivante
        nb_crois:     nombre de modèles que l'on veut croiser pour la génération suivante
        nb_mut:       nombre de modèles que l'on veut muter pour la génération suivante
    """
    
    # On commence par entraîner tout les modèles de la génération actuelle
    liste_models_trained = trainning_all_models(liste_models, nb_gene)
    liste_models.sort(key = lambda x: x.accuracy, reverse = True)
    

    save_results(liste_models_trained, nb_gene)


    # On va ensuite sélectionner les meilleurs modèles pour la génération suivante
    parents = selection(liste_models_trained, nb_select)

    # On va ensuite muter les modèles
    mutants = mutation(parents, nb_mut)

    # On va ensuite croiser les modèles

    new_generation = crossover(mutants + parents, nb_crois)
    next_gene = [model for model in new_generation]
    for model in mutants:
        next_gene.append(model)
    for model in parents:
        next_gene.append(model)

    # si on retombe sur plus de modèle que prévue, on capte le nombre pour éviter des problèmes ( si les calculs à l'initialisation on été mal fait en gros)    
    liste_models= liste_models[0:size_generation]
    
    return next_gene



def algo_genetique(nb_generations : int, nb_models : int, nb_couches : int, nb_select : int, nb_crois : int, nb_mut : int) -> list[Classifier]:
    """
        Fonction qui va permettre de faire tourner l'algo génétique
        nb_generations: nombre de générations que l'on veut faire tourner
        nb_models: nombre de modèles que l'on veut créer à chaque génération
        nb_select: nombre de modèles que l'on veut garder pour la génération suivante
        nb_crois: nombre de modèles que l'on veut croiser pour la génération suivante
        nb_mut: nombre de modèles que l'on veut muter pour la génération suivante
    """
    liste_models = gene_liste_models(nb_models, nb_couches)
    size_generation = len(liste_models)

    for i in range(nb_generations):
        liste_models = go_through_generation(liste_models, nb_select, nb_crois, nb_mut, i, size_generation)

    return liste_models


def algo_genetique_opti_model(nb_generations : int, model : Classifier, nb_select : int, nb_crois : int, nb_mut : int, nb_model : int) -> list[Classifier]:

    """"
        même fonction que juste au dessu sauf que l'on part d'un modele déjà existant que l'on va alors normalement optimiser
    
    """
    #On comme donc par faire muter notre modele jusqu'à optenir la premère population
    liste_model = [model]
    while len(liste_model)<50:
        liste_model += mutation(liste_model,len(liste_model))

    liste_model = liste_model[:50]

    for i in range(nb_generations):
        liste_models = go_through_generation(liste_model, nb_select, nb_crois, nb_mut, i)



    return liste_models

