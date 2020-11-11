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
# %matplotlib notebook


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

    parser.add_argument("-t", "--threshold", required=False, help="Seuil de détection (Par défaut 0.4")
    parser.add_argument("-i", "--img", required=True, help="chemin vers l'image")
    parser.add_argument("-m", "--mask", required=False, help="Chemin vers le masque")

    args = vars(parser.parse_args())

    img_test = args["img"]
    mask_test = args["mask"]
    threshold = 0.4 if args["threshold"] is None else float(args["threshold"])

    i_test = cv.imread(img_test)
    i_test = cv.cvtColor(i_test, cv.COLOR_BGR2RGB)
    i_mask = []
    if mask_test is not None:
        i_mask = cv.imread(mask_test, cv.IMREAD_GRAYSCALE)

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(2, 2, 1)
    plt.imshow(i_test)
    plt.title('image originale')
    plt.axis("off")

    fig.add_subplot(2, 2, 2)
    plt.imshow(detection_peau(i_test.copy(), Z_p, Z_n_p, RANGE, t_p, t_n_p, threshold, COLOR))
    plt.title('peau détecté')
    plt.axis("off")

    if mask_test is not None:
        fig.add_subplot(2, 2, 3)
        plt.imshow(i_mask, cmap='gray')
        plt.title('Masque')
        plt.axis("off")

        ax = plt.subplot(2, 2, 4)
        plt.imshow(peau_normale(i_test.copy(), i_mask, 100))
        plt.title('Résultat attendu')
        plt.axis("off")

    print(colored('Appuyer sur la touche 0 pour fermer l\'image', 'green'))
    print()

    plt.savefig('result.png')
    cv.imshow('Résultat de la détection', cv.resize(cv.imread('result.png') , (600, 600), interpolation=cv.INTER_AREA))
    cv.waitKey(0)
    cv.destroyAllWindows()