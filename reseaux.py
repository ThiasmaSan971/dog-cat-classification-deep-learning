
import numpy as np


class NeuralNetwork:

    def __init__(
        self,
        nb_entrees,
        nb_caches=64,
        taux_apprentissage=0.001
    ):
        self.taux_apprentissage = taux_apprentissage

        # Couche 1 : entrée -> couche cachée
        # Initialisation de He pour la fonction ReLU
        self.W1 = (
            np.random.randn(nb_entrees, nb_caches)
            * np.sqrt(2.0 / nb_entrees)
        ).astype(np.float32)

        self.b1 = np.zeros((1, nb_caches), dtype=np.float32)

        # Couche 2 : couche cachée -> sortie
        self.W2 = (
            np.random.randn(nb_caches, 1)
            * np.sqrt(1.0 / nb_caches)
        ).astype(np.float32)

        self.b2 = np.zeros((1, 1), dtype=np.float32)

    # --------------------------------------------------
    # Fonctions d'activation
    # --------------------------------------------------

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoide(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # --------------------------------------------------
    # Propagation avant
    # --------------------------------------------------

    def propagation_avant(self, X):

        if X.ndim != 2:
            raise ValueError(
                "X doit être une matrice 2D : "
                "(nombre_exemples, nombre_entrees)."
            )

        if X.shape[1] != self.W1.shape[0]:
            raise ValueError(
                f"Nombre d'entrées incorrect : {X.shape[1]}. "
                f"Le réseau en attend {self.W1.shape[0]}."
            )

        if not np.all(np.isfinite(X)):
            raise FloatingPointError(
                "Les entrées X contiennent des valeurs invalides."
            )

        # Couche cachée
        with np.errstate(over="ignore", invalid="ignore"):
            self.Z1 = X @ self.W1 + self.b1

        if not np.all(np.isfinite(self.Z1)):
            raise FloatingPointError(
                "Z1 invalide : vérifier X et les paramètres W1/b1. "
                f"Max |X| = {np.max(np.abs(X))}, "
                f"max |W1| = {np.max(np.abs(self.W1))}"
            )

        self.A1 = self.relu(self.Z1)

        # Couche de sortie
        with np.errstate(over="ignore", invalid="ignore"):
            self.Z2 = self.A1 @ self.W2 + self.b2

        if not np.all(np.isfinite(self.Z2)):
            raise FloatingPointError(
                "Z2 invalide : vérifier A1 et les paramètres W2/b2. "
                f"Max |A1| = {np.max(np.abs(self.A1))}, "
                f"max |W2| = {np.max(np.abs(self.W2))}"
            )

        self.A2 = self.sigmoide(self.Z2)

        if not np.all(np.isfinite(self.A2)):
            raise FloatingPointError(
                "Les prédictions contiennent des valeurs invalides."
            )

        return self.A2

    # --------------------------------------------------
    # Fonction de perte : entropie croisée binaire
    # --------------------------------------------------

    def calcul_perte(self, y, predictions):

        if y.shape != predictions.shape:
            raise ValueError(
                f"Formes incompatibles : y={y.shape}, "
                f"prédictions={predictions.shape}."
            )

        if not np.all(np.isfinite(y)):
            raise FloatingPointError(
                "Les étiquettes contiennent des valeurs invalides."
            )

        if not np.all(np.isfinite(predictions)):
            raise FloatingPointError(
                "Les prédictions contiennent des valeurs invalides."
            )

        epsilon = 1e-7

        predictions = np.clip(
            predictions,
            epsilon,
            1.0 - epsilon
        )

        perte = -np.mean(
            y * np.log(predictions)
            + (1.0 - y) * np.log(1.0 - predictions)
        )

        return float(perte)

    # --------------------------------------------------
    # Rétropropagation et descente de gradient
    # --------------------------------------------------

    def retropropagation(self, X, y):

        if X.shape[0] == 0:
            raise ValueError(
                "Impossible d'entraîner le réseau sur un lot vide."
            )

        if y.shape != (X.shape[0], 1):
            raise ValueError(
                f"Forme de y incorrecte : {y.shape}. "
                f"Forme attendue : ({X.shape[0]}, 1)."
            )

        if not np.all(np.isfinite(y)):
            raise FloatingPointError(
                "Les étiquettes contiennent des valeurs invalides."
            )

        if not np.all((y == 0) | (y == 1)):
            raise ValueError(
                "Les étiquettes doivent être 0 ou 1."
            )

        # Propagation avant
        predictions = self.propagation_avant(X)

        m = X.shape[0]

        # Gradient de la couche de sortie
        dZ2 = (predictions - y) / m

        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Gradient de la couche cachée
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Vérifier les gradients avant la mise à jour
        gradients = [
            ("dW1", dW1),
            ("db1", db1),
            ("dW2", dW2),
            ("db2", db2)
        ]

        for nom, gradient in gradients:
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError(
                    f"Gradient invalide détecté : {nom}."
                )

        # Calculer la perte avant la mise à jour
        perte = self.calcul_perte(y, predictions)

        # Calculer les nouveaux paramètres sans modifier
        # les paramètres actuels immédiatement
        eta = self.taux_apprentissage

        nouveaux_parametres = {
            "W1": self.W1 - eta * dW1,
            "b1": self.b1 - eta * db1,
            "W2": self.W2 - eta * dW2,
            "b2": self.b2 - eta * db2
        }

        # Vérifier les nouveaux paramètres
        for nom, parametre in nouveaux_parametres.items():
            if not np.all(np.isfinite(parametre)):
                raise FloatingPointError(
                    f"Le paramètre {nom} deviendrait invalide "
                    "après la mise à jour."
                )

        # Appliquer la mise à jour seulement si tout est valide
        self.W1 = nouveaux_parametres["W1"]
        self.b1 = nouveaux_parametres["b1"]
        self.W2 = nouveaux_parametres["W2"]
        self.b2 = nouveaux_parametres["b2"]

        return perte

    # --------------------------------------------------
    # Prédiction
    # --------------------------------------------------

    def predire(self, X):
        return self.propagation_avant(X)

    def predire_classes(self, X):
        probabilites = self.predire(X)
        return (probabilites >= 0.5).astype(np.int32)


# ------------------------------------------------------
# Test autonome du réseau
# Exécuté uniquement si ce fichier est lancé directement
# ------------------------------------------------------

if __name__ == "__main__":

    print("Test du fichier reseaux.py")

    np.random.seed(42)

    reseau = NeuralNetwork(nb_entrees=3072)

    X_test = np.zeros((2, 3072), dtype=np.float32)

    predictions = reseau.predire(X_test)

    print("Forme des prédictions :", predictions.shape)
    print("Prédictions :")
    print(predictions)
    print(
        "Toutes les prédictions sont finies :",
        np.all(np.isfinite(predictions))
    )

    assert predictions.shape == (2, 1)
    assert np.all(np.isfinite(predictions))
    assert np.allclose(predictions, 0.5)

    print("Test réussi.")
