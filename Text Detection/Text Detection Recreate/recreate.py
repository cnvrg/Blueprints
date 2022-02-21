import os
import sys
import argparse
import shutil
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="""Creator""")
parser.add_argument(
    "-f",
    "--images",
    action="store",
    dest="images",
    default="/data/images",
    required=True,
    help="""Location to the folder containing all training images""",
)
parser.add_argument(
    "--labels",
    action="store",
    dest="labels",
    default="/data/labels",
    help="""Location to the folder containing all training labels""",
)
args = parser.parse_args()


img_path = args.images
label_path = args.labels

images = [os.path.join(img_path, x) for x in os.listdir(img_path)]
annotations = [
    os.path.join(label_path, x) for x in os.listdir(label_path) if x[-3:] == "txt"
]

# run a sanity check on the first label file
try:
    to_check = annotations[0]
except:
    
    sys.exit("No file found with extension .txt in the labels folder.")
firstbox = 0
if not to_check.endswith(".txt"):  # confirm the file is .txt
    
    sys.exit("All label files have to be in the .txt file format")
with open(to_check, "r") as f:
    firstbox = f.readline().split(" ")
if len(firstbox) != 5:  # confirm the lenght of the first line in the label file is 5
    sys.exit(
        "Each line in the label file must only have 5 numbers. The class and the coordinates of the bounding box"
    )
if "." in firstbox[0]:  # confirm the first number is an integer
    
    sys.exit("The first number in each line of label file has to be integer.")
for numbers in firstbox[1:]:  # confirm the rest of the numbers are floats
    if "." not in numbers:
        
        sys.exit("The bounding box coordinates have to be floats")

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits
train_images, val_images, train_annotations, val_annotations = train_test_split(
    images, annotations, test_size=0.1, random_state=1
)
val_images, test_images, val_annotations, test_annotations = train_test_split(
    val_images, val_annotations, test_size=0.5, random_state=1
)

# create subdirectories of test/train/val for images folder
os.mkdir(os.path.join(img_path, "train"))
os.mkdir(os.path.join(img_path, "test"))
os.mkdir(os.path.join(img_path, "val"))

# create subdirectories of test/train/val for labels folder
os.mkdir(os.path.join(label_path, "train"))
os.mkdir(os.path.join(label_path, "test"))
os.mkdir(os.path.join(label_path, "val"))


# move the split files to the relevant folders
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


# Move the splits into their folders
move_files_to_folder(train_images, os.path.join(img_path, "train"))
move_files_to_folder(val_images, os.path.join(img_path, "val"))
move_files_to_folder(test_images, os.path.join(img_path, "test"))
move_files_to_folder(train_annotations, os.path.join(label_path, "train"))
move_files_to_folder(val_annotations, os.path.join(label_path, "val"))
move_files_to_folder(test_annotations, os.path.join(label_path, "test"))


# move the data files to cnvrg folder
shutil.move(img_path, "/cnvrg")
shutil.move(label_path, "/cnvrg")
