from fonctions_apprentissage import *


def evaluation_peau(a, b, Z_peau, Z_non_peau):
    p_peau = Z_peau[a, b]
    p_non_peau = Z_non_peau[a, b]
    return np.argmax([p_non_peau, p_peau])


def evaluation_peau_bayes(a, b, Z_peau, Z_non_peau, t_p, t_n_p):
    u = Z_peau[a, b] * (t_p / (t_p + t_n_p))
    p_peau = u / (u + (Z_non_peau[a, b] * (t_n_p / (t_p + t_n_p))))
    return p_peau


def peau_normale(img, masque, SEUIL):
    x, y = masque.shape

    for i in range(x):
        for j in range(y):
            if masque[i][j] < SEUIL:
                for k in range(0, 3):
                    img[i, j][k] = 0

    return img


def detection_peau(img, Z_peau, Z_non_peau, ECHELLE, t_p, t_n_p, seuil, COLOR):
    # On converti dans l'espace lab
    if COLOR == 0:
        img_lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    else:
        img_lab = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    # On modifie les intervalles de a et b
    temp = [0] * 256
    for i in range(256):
        temp[i] = floor(i / (256 / ECHELLE))

    h, w, d = img_lab.shape
    image = np.asarray(np.zeros((h, w, d), dtype=np.uint8))

    for i in range(h):
        for j in range(w):
            for k in range(1, 3):
                image[i, j][k] = temp[img_lab[i, j][k]]
            image[i, j][0] = img_lab[i, j][0]
    img_lab = image

    # On détecte la peau en mettant à 0 les pixels non peau
    for i in range(h):
        for j in range(w):
            if not evaluation_peau_bayes(img_lab[i, j][1], img_lab[i, j][2], Z_peau, Z_non_peau, t_p, t_n_p) >= seuil:
                for k in range(0, 3):
                    img[i, j][k] = 0

    return img


# Fonction d'évaluation
def evaluation(t_image, t_mask, Z_peau, Z_non_peau, total_peau, total_non_peau, SEUIL):
    true_positif = 0
    false_positif = 0
    true_negatif = 0
    false_negatif = 0

    for index, img in enumerate(t_image):
        mask = t_mask[index]
        x, y = mask.shape

        for i in range(x):
            for j in range(y):
                if mask[i][j] > 100:  # peau
                    if evaluation_peau_bayes(img[i, j][1], img[i, j][2], Z_peau, Z_non_peau, total_peau, total_non_peau) >= SEUIL:  # peau prédite
                        true_positif += 1
                    else:
                        false_negatif += 1
                else:  # non peau
                    if evaluation_peau_bayes(img[i, j][1], img[i, j][2], Z_peau, Z_non_peau, total_peau, total_non_peau) >= SEUIL:  # peau prédite
                        false_positif += 1
                    else:
                        true_negatif += 1

    precision = true_positif / (true_positif + false_positif)
    rappel = true_positif / (true_positif + false_negatif)
    f_score = (2 * precision * rappel) / (precision + rappel)

    return f_score, precision, rappel
