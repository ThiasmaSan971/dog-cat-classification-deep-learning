import numpy as np
from PIL import Image
from pathlib import Path

def lecture(chemin_du_fichier):
    try:
        with Image.open(chemin_du_fichier) as img:
            print("fichier {chemin_du_fichier} trouver")
            img_rgb = img.convert("RGB")
            img_redimensionnee = img_rgb.resize((150, 150), Image.Resampling.LANCZOS)
            matrice = np.array(img_redimensionnee)

            return matrice
    except FileNotFoundError:
        print("fichier {chemin_du_fichier} non trouver")
    except Exception as e:
        print(f"Erreur lors de la lecture de {chemin_du_fichier}: {e}")
        return None

def charger_et_preparer_dataset(chemin_du_dataset):
    chemin_dossier = Path(chemin_du_dataset)
    extensions_valides = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    liste_Images = []
    liste_etiquette = []

    #traitement des chats
    dossier_chat = chemin_dossier / "Cat"
    if dossier_chat.exists():
        for chemin_fichier in dossier_chat.iterdir():
            if chemin_fichier.is_file() and chemin_fichier.suffix.lower() in extensions_valides:
                #print(f"Traitement de l'image : {chemin_fichier.name}")
                img = lecture(chemin_fichier)
                if img is not None:
                    liste_Images.append(img)
                    # 0 pour les chats
                    liste_etiquette.append(0)
    #traitement des chiens
    dossier_chiens = chemin_dossier / "Dog"
    if dossier_chiens.exists():
        for chemin_fichier in dossier_chiens.iterdir():
            if chemin_fichier.is_file() and chemin_fichier.suffix.lower() in extensions_valides:
                #print(f"Traitement de l'image : {chemin_fichier.name}")
                img = lecture(chemin_fichier)
                if img is not None:
                    liste_Images.append(img)
                    # 1 pour les chiens
                    liste_etiquette.append(1)

    x = np.array(liste_Images) / 255.0
    y = np.array(liste_etiquette).reshape(-1,1)

    return separer_donnees(x, y, 0.8)

def separer_donnees(x, y, ratio_entrainement):
    nb_total = x.shape[0]
    
    indices = np.random.permutation(nb_total)
    
    x_melange = x[indices]
    y_melange = y[indices]

    limite = int(nb_total * ratio_entrainement)
    
    x_train = x_melange[:limite]
    y_train = y_melange[:limite]
    
    x_val = x_melange[limite:]
    y_val = y_melange[limite:]
    
    return x_train, y_train, x_val, y_val