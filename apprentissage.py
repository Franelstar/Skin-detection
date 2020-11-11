import cv2 as cv
import sys
import argparse

import matplotlib
import notebook as notebook
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
from math import *
import os
from mpl_toolkits import mplot3d
from fonctions_apprentissage import *
# %matplotlib notebook


def process_lab():
    print(colored('* Changement de l\'espace couleur de RGB à La*b*', attrs=['dark']))
    list_others_lab = []
    list_img_lab = rgb_to_lab(list_img)
    if list_other.shape[0] > 0:
        list_others_lab = rgb_to_lab(list_other)
    print(colored('* Changement de l\'espace couleur terminé', 'green'))
    print()

    print(colored('* Changement des dimensions des channels a et b', attrs=['dark']))
    list_img_lab = convert_lab_256_to_lab_x(list_img_lab, RANGE)
    if list_other.shape[0] > 0:
        list_others_lab = convert_lab_256_to_lab_x(list_others_lab, RANGE)
    print(colored('* Changement des dimensions terminé', 'green'))
    print()

    print(colored('* Calcul de l\'histogramme peau', attrs=['dark']))
    Z_peau, total_peau = histogramme_peau_lab(list_img_lab, list_mask, RANGE, THRESHOLD)
    print(colored('* Calcul de l\'histogramme peau terminé', 'green'))
    print()

    print(colored('* Calcul de l\'histogramme non peau', attrs=['dark']))
    if list_other.shape[0] == 0:
        Z_non_peau, total_non_peau = histogramme_non_peau_lab(list_img_lab, list_mask, RANGE, THRESHOLD)
    else:
        Z_non_peau, total_non_peau = histogramme_non_peau_lab_plus(list_img_lab, list_mask, list_others_lab, RANGE,
                                                               THRESHOLD)
    print(colored('* Calcul de l\'histogramme non peau terminé', 'green'))
    print()

    return Z_peau, Z_non_peau, total_peau, total_non_peau


def process_hsv():
    print(colored('* Changement de l\'espace couleur de RGB à HSV', attrs=['dark']))
    list_others_hsv = []
    list_img_hsv = rgb_to_hsv(list_img)
    if list_other.shape[0] > 0:
        list_others_hsv = rgb_to_hsv(list_other)
    print(colored('* Changement de l\'espace couleur terminé', 'green'))
    print()

    print(colored('* Changement des dimensions des channels h et s', attrs=['dark']))
    list_img_hsv = convert_hsv_256_to_hsv_x(list_img_hsv, RANGE)
    if list_other.shape[0] > 0:
        list_others_hsv = convert_hsv_256_to_hsv_x(list_others_hsv, RANGE)
    print(colored('* Changement des dimensions terminé', 'green'))
    print()

    print(colored('* Calcul de l\'histogramme peau', attrs=['dark']))
    Z_peau, total_peau = histogramme_peau_hsv(list_img_hsv, list_mask, RANGE, THRESHOLD)
    print(colored('* Calcul de l\'histogramme peau terminé', 'green'))
    print()

    print(colored('* Calcul de l\'histogramme non peau', attrs=['dark']))
    if list_other.shape[0] == 0:
        Z_non_peau, total_non_peau = histogramme_non_peau_hsv(list_img_hsv, list_mask, RANGE, THRESHOLD)
    else:
        Z_non_peau, total_non_peau = histogramme_non_peau_hsv_plus(list_img_hsv, list_mask, list_others_hsv, RANGE,
                                                                   THRESHOLD)
    print(colored('* Calcul de l\'histogramme non peau terminé', 'green'))
    print()

    return Z_peau, Z_non_peau, total_peau, total_non_peau


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--range", required=False, help="Interalle de valeurs des pixels (32 par défaut)", default=32)
    parser.add_argument("-t", "--img_train", required=True, help="Repertoire contenant les images d'apprentissage")
    parser.add_argument("-m", "--img_mask", required=True, help="Repertoire contenant les masques d'apprentissage")
    parser.add_argument("-a", "--img_complement", required=False, help="Repertoire contenant les images non peau pour "
                                                                       "completer les images d'entrainement")
    parser.add_argument("-l", "--lissage", required=False, help="Lissage des histogramme", action="store_true")
    parser.add_argument("-c", "--color", required=False, help="Espace de couleur", default='lab')

    args = vars(parser.parse_args())

    RANGE = 32 if args["range"] is None else int(args["range"])
    THRESHOLD = 100
    COLOR = 1 if args["color"] == 'hsv' or args["color"] == 'tsv' else 0

    if (int(RANGE) & (int(RANGE) - 1)) != 0:  # Verifier que c'est une puissance de 2
        print(colored('La valeur de "range" doit être un multiple de 2', 'red'))
        sys.exit(0)

    img_train = args["img_train"]
    img_mask = args["img_mask"]

    print(colored('* Chargement des images d\'entrainement', attrs=['dark']))
    list_img, list_mask, list_other = uplaod_image(img_train, img_mask, args["img_complement"])
    print(colored('* Chargement des images d\'entrainement terminé', 'green'))

    if list_img.shape[0] != list_mask.shape[0]:  # On vérifie que le nombre d'image correspond au nombre de mask
        print(colored('Le nombre d\'image et de masque ne corresponde pas', 'red'))
        sys.exit(0)

    print(colored('Nombre d\'images d\'entrainement: {}'.format(list_img.shape[0]), 'blue'), end="\n")
    print(colored('Nombre d\'images non peau complémentaire: {}'.format(list_other.shape[0]), 'blue'), end="\n")
    print()

    if COLOR == 0:
        Z_peau, Z_non_peau, total_peau, total_non_peau = process_lab()
    else:
        Z_peau, Z_non_peau, total_peau, total_non_peau = process_hsv()

    print(colored('Nombre total de pixel peau: {}'.format(int(total_peau)), 'blue'))
    print(colored('Nombre total de pixel non peau: {}'.format(int(total_non_peau)), 'blue'))
    print()

    if args["lissage"]:
        print(colored('* Lissage des histogrammes', attrs=['dark']))
        Z_peau = lissage(Z_peau)
        Z_non_peau = lissage(Z_non_peau)
        print(colored('* Lissage des histogrammes terminé', 'green'))
        print()

    print(colored('Appuyer sur la touche 0 pour fermer l\'image', 'green'))
    print()

    # Sauvegarde des parametres
    t_peau = np.array([total_peau, total_non_peau, RANGE, COLOR])
    np.savetxt('t_peau.txt', t_peau)
    np.savetxt('h_peau.txt', Z_peau)
    np.savetxt('h_non_peau.txt', Z_non_peau)

    if COLOR == 0:
        x = np.linspace(0, RANGE - 1, RANGE)
        y = np.linspace(0, RANGE - 1, RANGE)
    else:
        y = np.linspace(0, 180 - 1, 180)
        x = np.linspace(0, RANGE - 1, RANGE)

    X, Y = np.meshgrid(x, y)

    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z_peau, 500)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(50, 40)
    plt.savefig('hp.png')

    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z_non_peau, 500)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(50, 40)
    plt.savefig('hnp.png')

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(cv.imread('hp.png'), cv.COLOR_BGR2RGB))
    plt.title('Histogramme Peau')
    plt.axis("off")

    fig.add_subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(cv.imread('hnp.png'), cv.COLOR_BGR2RGB))
    plt.title('Histogramme Non Peau')
    plt.axis("off")

    plt.savefig('hist.png')
    cv.imshow('Appuyer sur la touche 0 pour fermer',
              cv.resize(cv.cvtColor(cv.imread('hist.png'), cv.COLOR_BGR2RGB), (1024, 512),
                        interpolation=cv.INTER_AREA))
    cv.waitKey(0)
    cv.destroyAllWindows()