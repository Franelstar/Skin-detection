# Skin detection based on color

## How to set up 

#### Reset the app

It allows to initialize the prediction files by default

Run `$ python reset.py`

#### Do a trainning

It allows you to perform a training session on a data set.

Run `$ python apprentissage.py -t original_training_folder -m mask_training_folder -a complement_training_folder`

the parameters are:

1. -t : path to the folder containing the original training images
1. -m : path to the folder containing the training masks
1. -a (optional) : path to the folder containing the additional training non-skin images (No mask)
1. -c (optional) : Choice of color space (lab or tsv). Default lab
1. -r (optional) : Choice of the number of colors of the channels (a and b for La*b*, t and s for TSV). Default 32
1. -l (optional) : To activate histogram smoothing. Default, not enabled

Exemple: 

`$ python apprentissage.py -t Dataset8_Abdomen/train/original_images/ -m Dataset8_Abdomen/train/skin_masks/ -a complement/ -c lab -r 32`

#### Make a detection

Detects skin pixels in an image.
The detection is made with the last model trained.

Run `$ python detect.py -i path_to_image -m mask_for_image -t threshold`

the parameters are:

1. -i: Path to image
1. -m (optional) : Path to the image mask
1. -t (optional) : Threshold value. Default 0.4

Exemple: 

`$ python detect.py -i Dataset8_Abdomen/test/original_images/01020.jpg -m Dataset8_Abdomen/test/skin_masks/01020.png -t 0.5`

#### Perform a model evaluation

The evaluation is made with the last model trained.

Run `$  python evaluation.py -i original_test_folder -m mask_test_folder -t threshold`

the parameters are:

1. -i: path to the folder containing the original test images
1. -m : path to the folder containing the mask test images
1. -t (optional) : Threshold value. Default 0.4

Exemple: 

`$ python evaluation.py -i Dataset8_Abdomen/test/original_images -m Dataset8_Abdomen/test/skin_masks -t 0.3`
