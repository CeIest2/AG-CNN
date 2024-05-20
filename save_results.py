from CNN_classe import Classifier


def save_results(liste_models : list[Classifier], nb_gene : int) -> None:
    """
        Fonction qui va permettre de sauvegarder les résultats de la génération actuelle
        liste_models: liste des modèles de la génération actuelle
    """
    with open("results.txt", "a") as file:
        file.write("##############################################################\n")
        file.write(f"Génération {nb_gene}\n")
        for model in liste_models:
            file.write(f"{model.para} | {model.accuracy} | {model.count_parameters()} | {model.time_training}\n")