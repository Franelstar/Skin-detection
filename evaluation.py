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
from fonctions_prediction_evaluation import *

if __name__ == '__main__':

    try:
        Z_p = np.loadtxt('h_peau.txt')
        Z_n_p = np.loadtxt('h_non_peau.txt')
        t_peau = np.loadtxt('t_peau.txt')
    except:
        print(colored('Vous devez d\'abord effectuer l\'apprentissage ou initialiser l\'application', 'red'))
        print(colored('Referez-vous à la documentation', 'red'))
        sys.exit(0)

    RANGE = t_peau[2]
    t_p = t_peau[0]
    t_n_p = t_peau[1]
    COLOR = t_peau[3]

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--THRESHOLD", required=False, help="Seuil de détection (Par défaut 0.4)")
    parser.add_argument("-i", "--img_test", required=True, help="Repertoire contenant les images de test")
    parser.add_argument("-m", "--img_mask", required=True, help="Repertoire contenant les masques de test")

    args = vars(parser.parse_args())

    THRESHOLD = 0.4 if args["THRESHOLD"] is None else args["THRESHOLD"]

    img_test = args["img_test"]
    img_mask = args["img_mask"]

    print(colored('* Chargement des images de test', attrs=['dark']))
    list_img, list_mask, list_other = uplaod_image(img_test, img_mask, None)
    print(colored('* Chargement des images de test terminé', 'green'))
    print()

    if list_img.shape[0] != list_mask.shape[0]: # On vérifie que le nombre d'image correspond au nombre de mask
        print(colored('Le nombre d\'image et de masque ne corresponde pas', 'red'))
        sys.exit(0)

    print(colored('Nombre d\'images de test: {}'.format(list_img.shape[0]), 'blue'), end="\n")
    print()

    if COLOR == 0:
        print(colored('* Changement de l\'espace couleur de RGB à La*b*', attrs=['dark']))
        list_img_c = rgb_to_lab(list_img)
    else:
        print(colored('* Changement de l\'espace couleur de RGB à HSV', attrs=['dark']))
        list_img_c = rgb_to_hsv(list_img)
    print(colored('* Changement de l\'espace couleur terminé', 'green'))
    print()

    if COLOR == 0:
        print(colored('* Changement des dimentions des channels a et b', attrs=['dark']))
        list_img_c = convert_lab_256_to_lab_x(list_img_c, RANGE)
    else:
        print(colored('* Changement des dimentions des channels h et s', attrs=['dark']))
        list_img_c = convert_hsv_256_to_hsv_x(list_img_c, RANGE)
    print(colored('* Changement des dimentions terminé', 'green'))
    print()

    print(colored('* Début de l\'évaluation', attrs=['dark']))
    f_score, precision, rappel = evaluation(list_img_c, list_mask, Z_p, Z_n_p, t_p, t_n_p, float(THRESHOLD))
    print(colored('*** Evaluation terminée ***', 'blue'))
    print(colored('*** Précision : {} %'.format(round(precision * 100, 2)), 'green'))
    print(colored('*** Rappel : {} %'.format(round(rappel * 100, 2)), 'green'))
    print(colored('*** F-Score : {} %'.format(round(f_score * 100, 2)), 'green'))