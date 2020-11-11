import shutil
import os
from termcolor import colored
import sys

path_file = "modele"
list_files = [f for f in os.listdir(path_file) if os.path.isfile(os.path.join(path_file, f))]

if len(list_files) == 0:
    print(colored('Cette action n\'est pas possible, vous devez faire un apprentissage', 'red'))
    print(colored('Referez-vous à la documentation', 'red'))
    sys.exit(0)

for f in list_files:
    shutil.copyfile('modele/' + f, './' + f)

print()
print(colored('Application réinitialisé, vous pouvez effectuer des prédictions', 'green'))
print()
