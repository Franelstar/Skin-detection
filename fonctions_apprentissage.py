import os
import cv2 as cv
import numpy as np
from math import *


# upload images and masks
def uplaod_image(train_path, train_mask, other_path):
    path_original_img_train = train_path
    path_masks_img_train = train_mask

    list_images = [f for f in os.listdir(path_original_img_train) if
                   os.path.isfile(os.path.join(path_original_img_train, f))]

    t_list_images_originals = []
    t_list_images_masks = []
    t_list_images_other = []

    for img in list_images:
        image = cv.imread(os.path.join(path_original_img_train, img))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        t_list_images_originals.append(image)

    for img in list_images:
        image_mask = cv.imread(os.path.join(path_masks_img_train, img.split('.')[0] + '.png'), cv.IMREAD_GRAYSCALE)
        t_list_images_masks.append(image_mask)

    if other_path is not None:
        path_other_img_train = other_path
        list_images = [f for f in os.listdir(path_other_img_train) if
                       os.path.isfile(os.path.join(path_other_img_train, f))]
        for img in list_images:
            image = cv.imread(os.path.join(path_other_img_train, img))
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            t_list_images_other.append(image)

    return np.array(t_list_images_originals), np.array(t_list_images_masks), np.array(t_list_images_other)


# On change l'espace de couleur de RGB à Lab
def rgb_to_lab(t_rgb):
    t_lab = [0] * t_rgb.shape[0]
    for i, img in enumerate(t_rgb):
        t_lab[i] = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    return np.array(t_lab)


# On change l'espace de couleur de RGB à HSV
def rgb_to_hsv(t_rgb):
    t_lab = [0] * t_rgb.shape[0]
    for i, img in enumerate(t_rgb):
        t_lab[i] = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    return np.array(t_lab)


# On convertie l'intervalle dans lequel les pixels prennent leurs valeurs pour les dimensions a et b
def convert_lab_256_to_lab_x(t_img, ECHELLE):
    temp = [0] * 256
    for i in range(256):
        temp[i] = floor(i / (256 / ECHELLE))

    for index, img in enumerate(t_img):
        h, w, d = img.shape
        image = np.asarray(np.zeros((h, w, d), dtype=np.uint8))

        for i in range(h):
            for j in range(w):
                for k in range(1, 3):
                    image[i, j][k] = temp[img[i, j][k]]
                image[i, j][0] = img[i, j][0]
        t_img[index] = image

    return t_img


# On convertie l'intervalle dans lequel les pixels prennent leurs valeurs pour les dimensions s et v
def convert_hsv_256_to_hsv_x(t_img, ECHELLE):
    temp = [0] * 256
    for i in range(256):
        temp[i] = floor(i / (256 / ECHELLE))

    for index, img in enumerate(t_img):
        h, w, d = img.shape
        image = np.asarray(np.zeros((h, w, d), dtype=np.uint8))

        for i in range(h):
            for j in range(w):
                image[i, j][1] = temp[img[i, j][1]]
                image[i, j][0] = img[i, j][0]
                image[i, j][2] = temp[img[i, j][2]]
        t_img[index] = image

    return t_img


# fonction pour calculer l'histogramme peau dans l'espace de couleur Lab
def histogramme_peau_lab(t_images_lab, t_masque, ECHELLE, SEUIL):
    z = np.zeros((ECHELLE, ECHELLE))

    for index, img in enumerate(t_images_lab):
        l_channel, a_channel, b_channel = cv.split(img)
        masque = t_masque[index]

        x, y = a_channel.shape

        for i in range(x):
            for j in range(y):
                if masque[i][j] > SEUIL:
                    x_ = a_channel[i, j]
                    y_ = b_channel[i, j]
                    z[x_, y_] += 1
    return z / sum(sum(z)), sum(sum(z))


# fonction pour calculer l'histogramme peau dans l'espace de couleur HSV
def histogramme_peau_hsv(t_images_hsv, t_masque, ECHELLE, SEUIL):
    z = np.zeros((180, ECHELLE))

    for index, img in enumerate(t_images_hsv):
        s_channel, h_channel, v_channel = cv.split(img)
        masque = t_masque[index]

        x, y = s_channel.shape

        for i in range(x):
            for j in range(y):
                if masque[i][j] > SEUIL:
                    x_ = s_channel[i, j]
                    y_ = h_channel[i, j]
                    z[x_, y_] += 1
    return z / sum(sum(z)), sum(sum(z))


# fonction pour calculer l'histogramme non peau dans l'espace Lab
def histogramme_non_peau_lab(t_images_lab, t_masque, ECHELLE, SEUIL):
    z = np.zeros((ECHELLE, ECHELLE))

    for index, img in enumerate(t_images_lab):
        l_channel, a_channel, b_channel = cv.split(img)
        masque = t_masque[index]

        x, y = a_channel.shape

        for i in range(x):
            for j in range(y):
                if masque[i][j] < SEUIL:
                    x_ = a_channel[i, j]
                    y_ = b_channel[i, j]
                    z[x_, y_] += 1
    return z / sum(sum(z)), sum(sum(z))


# fonction pour calculer l'histogramme non peau dans l'espace HSV
def histogramme_non_peau_hsv(t_images_hsv, t_masque, ECHELLE, SEUIL):
    z = np.zeros((180, ECHELLE))

    for index, img in enumerate(t_images_hsv):
        s_channel, h_channel, v_channel = cv.split(img)
        masque = t_masque[index]

        x, y = s_channel.shape

        for i in range(x):
            for j in range(y):
                if masque[i][j] < SEUIL:
                    x_ = s_channel[i, j]
                    y_ = h_channel[i, j]
                    z[x_, y_] += 1
    return z / sum(sum(z)), sum(sum(z))


# fonction pour augmenter le nombre de pixel dans l'histogramme non peau Lab
def histogramme_non_peau_lab_plus(t_images_lab, t_masque, t_autres, ECHELLE, SEUIL):
    z = np.zeros((ECHELLE, ECHELLE))

    for index, img in enumerate(t_images_lab):
        l_channel, a_channel, b_channel = cv.split(img)
        masque = t_masque[index]

        x, y = a_channel.shape

        for i in range(x):
            for j in range(y):
                if masque[i][j] < SEUIL:
                    x_ = a_channel[i, j]
                    y_ = b_channel[i, j]
                    z[x_, y_] += 1

    for index, img in enumerate(t_autres):
        l_channel, a_channel, b_channel = cv.split(img)

        x, y = a_channel.shape

        for i in range(x):
            for j in range(y):
                x_ = a_channel[i, j]
                y_ = b_channel[i, j]
                z[x_, y_] += 1

    return z / sum(sum(z)), sum(sum(z))


# fonction pour augmenter le nombre de pixel dans l'histogramme non peau HSV
def histogramme_non_peau_hsv_plus(t_images_lab, t_masque, t_autres, ECHELLE, SEUIL):
    z = np.zeros((180, ECHELLE))

    for index, img in enumerate(t_images_lab):
        h_channel, s_channel, v_channel = cv.split(img)
        masque = t_masque[index]

        x, y = h_channel.shape

        for i in range(x):
            for j in range(y):
                if masque[i][j] < SEUIL:
                    x_ = h_channel[i, j]
                    y_ = s_channel[i, j]
                    z[x_, y_] += 1

    for index, img in enumerate(t_autres):
        h_channel, s_channel, v_channel = cv.split(img)

        x, y = h_channel.shape

        for i in range(x):
            for j in range(y):
                x_ = h_channel[i, j]
                y_ = s_channel[i, j]
                z[x_, y_] += 1

    return z / sum(sum(z)), sum(sum(z))


# Fonction de lissage des histogrammes
def lissage(image):
    result = np.zeros(image.shape)

    x, y = image.shape

    for i in range(x):
        for j in range(y):
            somme = 0
            compteur = 0
            if i > 0:
                if j > 0:
                    somme += image[i - 1, j - 1]
                    compteur += 1
                somme += image[i - 1, j]
                compteur += 1
                if j < y - 1:
                    somme += image[i - 1, j + 1]
                    compteur += 1
            if j > 0:
                somme += image[i, j - 1]
                compteur += 1
            if j < y - 1:
                somme += image[i, j + 1]
                compteur += 1
            if i < x - 1:
                if j > 0:
                    somme += image[i + 1, j - 1]
                    compteur += 1
                somme += image[i + 1, j]
                compteur += 1
                if j < y - 1:
                    somme += image[i + 1, j + 1]
                    compteur += 1
            result[i, j] = image[i, j] + (somme / compteur)
    return result