import numpy as np
import pickle
from pathlib import Path

from donnees import charger_et_preparer_dataset
from reseaux import NeuralNetwork


def entrainer(
    modele,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=40,
    taille_lot=64
):
    # Aplatir les images : (m, hauteur, largeur, canaux)
    # devient (m, nombre_de_pixels_et_canaux)
    x_train = x_train.reshape(len(x_train), -1).astype(np.float32)
    x_val = x_val.reshape(len(x_val), -1).astype(np.float32)

    y_train = y_train.reshape(-1, 1).astype(np.float32)
    y_val = y_val.reshape(-1, 1).astype(np.float32)

    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("Les ensembles train et validation doivent être non vides.")

    if taille_lot <= 0:
        raise ValueError("La taille d'un mini-lot doit être positive.")

    historique = {
        "perte_train": [],
        "accuracy_train": [],
        "perte_val": [],
        "accuracy_val": []
    }

    for epoch in range(epochs):

        # Mélanger les exemples à chaque époque
        indices = np.random.permutation(len(x_train))

        x_melange = x_train[indices]
        y_melange = y_train[indices]

        # Apprentissage par mini-lots
        for debut in range(0, len(x_train), taille_lot):
            fin = debut + taille_lot

            X_batch = x_melange[debut:fin]
            y_batch = y_melange[debut:fin]

            modele.retropropagation(X_batch, y_batch)

        # Évaluation sur l'ensemble d'entraînement
        pred_train = modele.predire(x_train)

        perte_train = modele.calcul_perte(y_train, pred_train)

        acc_train = np.mean(
            (pred_train >= 0.5) == (y_train >= 0.5)
        )

        # Évaluation sur l'ensemble de validation
        pred_val = modele.predire(x_val)

        perte_val = modele.calcul_perte(y_val, pred_val)

        acc_val = np.mean(
            (pred_val >= 0.5) == (y_val >= 0.5)
        )

        # Enregistrer les résultats
        historique["perte_train"].append(float(perte_train))
        historique["accuracy_train"].append(float(acc_train))
        historique["perte_val"].append(float(perte_val))
        historique["accuracy_val"].append(float(acc_val))

        print(
            f"Époque {epoch + 1}/{epochs} | "
            f"Perte train : {perte_train:.4f} | "
            f"Accuracy train : {acc_train:.2%} | "
            f"Perte val : {perte_val:.4f} | "
            f"Accuracy val : {acc_val:.2%}"
        )
        modele.taux_apprentissage *= 0.95

    return historique


def predire(modele, image):
    """
    Prédit la classe d'une image RGB brute,
    dont les pixels sont compris entre 0 et 255.
    """

    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être une matrice RGB à trois canaux.")

    # Même normalisation que dans donnees.py
    X = image / 255.0

    # Aplatir l'image pour le réseau
    X = X.reshape(1, -1)

    probabilite_chien = float(modele.predire(X)[0, 0])

    if probabilite_chien >= 0.5:
        classe = "Chien"
    else:
        classe = "Chat"

    return classe, probabilite_chien


if __name__ == "__main__":

    chemin_dataset = (
        "/Users/matthiascoudoux/Documents/Master_Informatique/"
        "Master_2/Projet_IA/projet_Chien:chat/Dataset/PetImages"
    )

    resolution = 32

    x_train, y_train, x_val, y_val = charger_et_preparer_dataset(
        chemin_dataset,
        resolution=resolution
    )

    modele = NeuralNetwork(
        nb_entrees=resolution * resolution * 3,
        nb_caches=64,
        taux_apprentissage=0.001
    )

    historique = entrainer(
        modele,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=40,
        taille_lot=64
    )

    # =====================================================================
    # 💾 SAUVEGARDE DU MODÈLE EN FIN D'ENTRAÎNEMENT
    # =====================================================================
    # Définition d'un chemin absolu direct dans votre dossier de projet Master 2
    chemin_dossier_projet = "/Users/matthiascoudoux/Documents/Master_Informatique/Master_2/Projet_IA/projet_Chien:chat"
    chemin_sauvegarde = Path(chemin_dossier_projet) / "ia_agent_chat_chien.pkl"

    print("\n💾 Sauvegarde du modèle en cours...")
    try:
        with open(chemin_sauvegarde, "wb") as f:
            pickle.dump(modele, f)
        print(f"🧠 Succès ! Le cerveau de votre agent est sauvegardé dans : {chemin_sauvegarde}")
        print("Toutes les matrices de poids sont sécurisées sur votre disque dur.")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du fichier .pkl : {e}")
