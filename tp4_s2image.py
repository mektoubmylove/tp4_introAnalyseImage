
import random

import cv2
import numpy as np



def parcoursCC(image, p, label, labels_image):
    """
    Parcours en profondeur pour l'étiquetage des composantes connexes.
    """
    # Initialisation de la pile
    stack = [p]
    # Marquer le pixel comme visité en l'étiquetant
    labels_image[p[1], p[0]] = label
    size=1 #nombre de pixels de la composante

    # Directions pour l'adjacence 4 (haut, bas, gauche, droite)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Tant que la pile n'est pas vide, on continue l'exploration
    while stack:
        current_pixel = stack.pop()

        # Parcourir les voisins du pixel courant
        for direction in directions:
            voisin_x = current_pixel[0] + direction[0]
            voisin_y = current_pixel[1] + direction[1]

            # on verif si le voisin est dans l'image et si c'est un pixel non visité et un pixel de valeur 1
            if (0 <= voisin_x < image.shape[1] and 0 <= voisin_y < image.shape[0]
                    and image[voisin_y, voisin_x] == 1 and labels_image[voisin_y, voisin_x] == 0):
                # Marquer ce pixel comme visité et l'ajouter à la pile
                labels_image[voisin_y, voisin_x] = label
                stack.append((voisin_x, voisin_y))
                size+=1
    return size

def ccLabel(image_path):
    """
    Fonction d'étiquetage des composantes connexes dans une image binaire.
    Chaque composante connexe reçoit une couleur aléatoire.
    """
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Vérifier si l'image est bien chargée
    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin.")

    # Binarisr l'image (128 -> valeur arbitraire)
    image = np.where(image > 128, 1, 0).astype(np.uint8)

    # Créer une image pour les labels
    labels_image = np.zeros_like(image, dtype=np.int32)

    # Créer une image en couleur pour afficher les résultats
    colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)


    label = 1  # Compteur pour identifier chaque composante connexe

    # Parcourir tous les pixels de l'image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Si le pixel est un pixel blanc et n'a pas été étiqueté
            if image[y, x] == 1 and labels_image[y, x] == 0:
                # Appeler la fonction parcoursCC pour marquer tous les pixels connectés
                parcoursCC(image, (x, y), label, labels_image)

                # on assigne une couleur aléatoire à cette composante connexe
                color = [random.randint(0, 255) for c in range(3)]

                # Colorer les pixels de la composante connexe
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        if labels_image[i, j] == label:
                            colored_image[i, j] = color

                label += 1  # Incrémenter le label pour la prochaine composante connexe


    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original", 512, 512)
    cv2.imshow("Original", image)

    cv2.namedWindow("Étiqueté", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Étiqueté", 512, 512)
    cv2.imshow("Étiqueté", colored_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#ccLabel("C:/Users/rhogu/Pictures/delon.jpg")

def ccAreaFilter(image_path, size):
    """
    Filtre d’aire : supprime les composantes connexes de taille < size.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin.")

    binary_image = np.where(image > 128, 1, 0).astype(np.uint8)

    #image pour stocker les etiquettes des composantes
    labels_image = np.zeros_like(binary_image, dtype=np.int32)
    taille_composantes = {}  # Dictionnaire pour stocker la taille des composantes

    label = 1 # Compteur pour identifier chaque composante connexe

    # Parcourir tous les pixels de l'image
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 1 and labels_image[y, x] == 0:
                #trouver la composante et recup sa taille
                taille_compo = parcoursCC(binary_image, (x, y), label, labels_image)

                # Enregistrer la taille de la composante
                if taille_compo is not None:
                    taille_composantes[label] = taille_compo
                    label += 1 #on passe au label suivant

    # Image filtrée où seules les composantes de taille suffisante sont conservées
    filtered_image = np.zeros_like(binary_image, dtype=np.uint8)

    #parcourir l'image avec les etuqttes pour filtrer les composantes < size
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            label_value = labels_image[y, x]

            # le label est valide et taille est suffisante
            if label_value > 0 and label_value in taille_composantes:
                if taille_composantes[label_value] >= size:
                    filtered_image[y, x] = 255  # Conserver le pixel dans l'image filtrée


    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original", 512, 512)
    cv2.imshow("Original", binary_image * 255)

    cv2.namedWindow("Filtrée", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Filtrée", 512, 512)
    cv2.imshow("Filtrée", filtered_image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


#ccAreaFilter("C:/Users/rhogu/Pictures/delon.jpg",100)
def find(label, representant):
    """Trouve le représentant du label ."""
    while representant[label] != label:  # Tant que le label n'est pas son propre représentant
        representant[label] = representant[representant[label]] #on compresse le chemin
        label = representant[label]
    return label

def union(label1, label2, representant):
    """"Fusionne deux ensembles de labels."""
    rpzfinal1 = find(label1, representant) # Trouver le représentant final du premier label
    rpzfinal2 = find(label2, representant) # Trouver le représentant final du second label
    if rpzfinal1 != rpzfinal2: # Si les représentants sont différents, les fusionner
        representant[rpzfinal2] = rpzfinal1  # Faire pointer le représentant du second vers le premier

def first_pass(binary_image):
    """Première passe : Attribution des labels et gestion des équivalences."""
    height, width = binary_image.shape
    labels = np.zeros((height, width), dtype=int)
    parent = {}
    next_label = 1  # Premier label disponible

    for y in range(height):
        for x in range(width):
            if binary_image[y, x] == 1:
                voisins = []

                # Vérifier les voisins Nord et Ouest
                if x > 0 and labels[y, x - 1] > 0:  # Ouest
                    voisins.append(labels[y, x - 1])
                if y > 0 and labels[y - 1, x] > 0:  # Nord
                    voisins.append(labels[y - 1, x])

                if not voisins:  # Aucun voisin labellisé -> Nouveau label
                    labels[y, x] = next_label # Attribution d'un nouveau label
                    parent[next_label] = next_label  # Init Union-Find
                    next_label += 1
                else:  # Il y a des voisins labellisés
                    min_label = min(voisins)
                    labels[y, x] = min_label

                    # Fusionner les labels équivalents
                    for label in voisins:
                        union(min_label, label, parent)

    return labels, parent

def second_pass(labels, parent):
    """Deuxième passe : Attribution du label final """
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            if labels[y, x] > 0: # Si le pixel appartient à une composante
                labels[y, x] = find(labels[y, x], parent) # Trouver son représentant final
    return labels

def colore_labels(labels):
    height, width = labels.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)
    random_colors = {}

    for y in range(height):
        for x in range(width):
            label_value = labels[y, x]
            if label_value > 0:
                if label_value not in random_colors:
                    random_colors[label_value] = [np.random.randint(0, 255) for c in range(3)]
                colored_image[y, x] = random_colors[label_value]

    return colored_image

def ccTwoPassLabel(image_path):
    """
    Implémentation de l'algorithme de labélisation en 2 passes (Hoshen-Kopelman).
    """

    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Erreur de chargement de l'image : {image_path}")

    binary_image = np.where(image > 128, 1, 0).astype(np.uint8)

    # Première passe : Attribution des labels et gestion des équivalences
    labels, parent = first_pass(binary_image)

    # Deuxième passe : Correction des labels via Union-Find
    final_labels = second_pass(labels, parent)

    colored_img = colore_labels(final_labels)

    # Affichage des résultats
    cv2.imshow("Original", binary_image * 255)
    cv2.imshow("2pass (Colored)", colored_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#ccTwoPassLabel("C:/Users/rhogu/Pictures/delon.jpg")

def tresholdOtsu(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Impossible de charger l'image. Vérifiez le chemin du fichier.")

    # Dimensions de l'image
    height, width = image.shape
    total_pixels = height * width

    # Initialisation de l'histogramme des niveaux de gris
    hist = [0] * 256
    for i in range(height):
        for j in range(width):
            intensity = int(image[i, j])
            hist[intensity] += 1         # Incrémentation de la valeur du pixel dans l'histogramme

    # Normalisation de l'histogramme pour obtenir les probabilités d'apparition des intensités
    prob = [0] * 256
    for i in range(256):
        prob[i] = hist[i] / total_pixels

    best_threshold = 0 #meilleur seuil
    max_variance = 0  # Variance inter-classes maximale

    # Parcourir tous les seuils possibles
    for t in range(256):
        # Calcul des poids des deux classes (pixels resp. inférieurs et supérieurs >= au seuil)
        w0 = sum(prob[:t])
        w1 = sum(prob[t:])

        # Éviter les divisions par zéro
        if w0 == 0 or w1 == 0:
            continue

        # Calculer les moyennes des intensités des deux classes
        mean0 = sum(i * prob[i] for i in range(t)) / w0
        mean1 = sum(i * prob[i] for i in range(t, 256)) / w1

        # Calculer la variance inter-classe
        variance = w0 * w1 * (mean0 - mean1) ** 2

        # Mettre à jour le meilleur seuil si une variance plus grande est trouvée
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    # Création d'une image binaire avec le seuil optimal trouvé
    binary_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if image[i, j] >= best_threshold:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0

    combined_image = cv2.hconcat([image, binary_image])


    cv2.namedWindow("Original and binary", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original and binary", 1024, 576)
    cv2.imshow("Original and binary", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(best_threshold)

#tresholdOtsu("C:/Users/rhogu/Pictures/cantouno.jpg")