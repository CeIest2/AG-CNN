from algo_genetique import algo_genetique, algo_genetique_opti_model
from CNN_classe import Classifier


def main():
    nb_generations = 70
    nb_models = 100
    nb_couches = 2
    nb_select = 25
    nb_crois = 25
    nb_mut = 50

    best_model = algo_genetique(nb_generations, nb_models, nb_couches, nb_select, nb_crois, nb_mut)
    print("fin de la simulation")
    return 0


def algo_autour_model(model : Classifier):
    """
        Au lieu de faire un simulation qui par entièrement de modèles randoms, on va partir d'un modèle existant,
        le faire muter et l'optimiser seulement avec l'algo génétique, 
    
    """
    nb_generations = 25
    nb_model = 50
    nb_select = 15
    nb_crois = 20
    nb_mut = 15

    list_model = algo_genetique_opti_model(nb_generations, model, nb_select, nb_crois, nb_mut, nb_model)


main()
"""
model = Classifier([[32, 3, 0.2], [64, 3, 0.2],[128, 3, 0.2],[256, 3, 0.2]])
algo_autour_model(model)

"""