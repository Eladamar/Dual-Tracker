import os
import glob
import numpy as np

from PIL import Image
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator



"""
Convert VisDrone annotations to YOLOv3(coco) annotations
"""

source_directory = './annotations'

image_filenames = glob.glob(os.path.join('./images', '*.jpg'))
Path('./labels').mkdir(parents=True, exist_ok=True)
Path('./annotated').mkdir(parents=True, exist_ok=True)

new_mapped_cat = {
    '1': '0', # pedestrian to person
    '2': '0', # person to person
    '4': '1', # car to car
    '5': '1', # van to car
    '3': '2', # bicycle to bycicle
    '10': '3', # motor to motor
    '9': '4', # bus to bus
    '6': '5' , #truck to truck
    '0': 'ignore', # ignore to ignore
    '7': 'ignore', # tricycle to ignore
    '8': 'ignore', # awning-tricycle to ignore
    '11': 'ignore' # others to ignore
}

new_cat_names = {
    '0': "person",
    '1': "car",
    '2': "bicycle",
    '3': "motor",
    '4': "bus",
    '5': "truck"
}

bbox_colors = {
    '0': "grey",
    '1': "red",
    '2': "blue",
    '3': "#0099FF",
    '4': "violet",
    '5': "#EB70AA"
}


for i, img_fname in enumerate(image_filenames):
    annotation_fname = img_fname.split('/')[-1]
    annotation_fname = annotation_fname.replace('.jpg', '.txt')
    my_file = Path(f"./labels/{annotation_fname}")
    if my_file.is_file():
        continue
    im = Image.open(img_fname)
    width, height = im.size
    # read the initial lines in the txt
    with open(f"./annotations/{annotation_fname}", 'r') as f:
        annotations = f.read().strip().split('\n')
    # split them on commas ','
    annotations = [x.split(',') for x in annotations]
    if i % 100 == 0:
        # Create plot
        plt.figure()
        fig, ax = plt.subplots(1)
        img = np.array(im)
        ax.imshow(img)

    with open(f"./labels/{annotation_fname}", 'w') as f:
        for ann in annotations:
            xmin, ymin, xwidth, yheight, score, cat, trunc, occ = ann
            x_ctr = (int(xmin) + (int(xwidth) / 2)) / width
            y_ctr = (int(ymin) + (int(yheight) / 2)) / height
            new_height = (int(yheight)/float(height))
            new_width = (int(xwidth) / float(width))
            new_cat = new_mapped_cat[cat]

            if score == 0: continue
            if new_mapped_cat[cat] == "ignore": continue
            f.write('{cat} {x_ctr:.6f} {y_ctr:.6f} {width:.6f} {height:.6f}\n'.format(cat=new_cat, x_ctr=x_ctr, y_ctr=y_ctr, width=new_width, height=new_height))
            if i % 100 == 0:
                color = bbox_colors[new_cat]
                # Create a Rectangle patch
                bbox = patches.Rectangle((float(xmin), float(ymin)), new_width * float(width), new_height * float(height), linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    float(xmin),
                    float(ymin),
                    s=new_cat_names[new_cat],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
    if i % 100 == 0:
        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        annotated_image = img_fname.split('/')[-1].replace('.jpg', '.png')
        plt.savefig(f"./annotated/{annotated_image}", bbox_inches="tight", pad_inches=0.0)
        plt.close('all')
        print('.',end='')