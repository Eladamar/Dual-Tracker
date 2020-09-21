import os
import cv2
import glob
from pathlib import Path
import pandas as pd
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import numpy as np

"""convert sdd videos to images and create YOLOv3 annotations"""

path_out = "./data/stanford/frames"
stanford_areas = glob.glob("./data/stanford/videos/*/")
annotations_areas = glob.glob("./data/stanford/annotations/*/")
annotations_out = './data/stanford/labels'
samples_out = "./data/stanford/samples"

FPS = 30
logger_file = './data/stanford/images_names'

Sdd2Coco = {
    "Pedestrian": 0,
    "Biker": 2,
    "Car": 1,
    "Bus": 4,
    "Skater": 0
}

cat_names = {
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

def convert_video(vid, area_output, area_name, videoi, logger_file_fd):
    cap = cv2.VideoCapture(vid)

    count = 0
    success = True
    while success:
        success, image = cap.read()
        if count % FPS == 0:
            name_to_save = f"{area_name}_{videoi}_frame{count}.png"
            path = os.path.join(area_output, name_to_save)
            if not os.path.isfile(path):
                cv2.imwrite(path, image)
                logger_file_fd.write(name_to_save + '\n')
        count += 1

def video_2_image(logger_file_fd):
    for area in stanford_areas:
        area_name = area[:-1].split('/')[-1]
        area_output = os.path.join(path_out, area_name)
        Path(area_output).mkdir(parents=True, exist_ok=True)
        for i, videoi in enumerate(os.listdir(area)):
            vid = os.path.join(area + videoi, 'video.mov')
            convert_video(vid, area_output, area_name, videoi, logger_file_fd)



def create_annotations():
    for area in annotations_areas:
        area_name = area[:-1].split('/')[-1]
        area_output = os.path.join(annotations_out, area_name)
        frames_dir = os.path.join(path_out, area_name)
        Path(area_output).mkdir(parents=True, exist_ok=True)
        for i, videoi in enumerate(os.listdir(area)):

            annotation = os.path.join(area + videoi, 'annotations.txt')
            df_annotation = pd.read_csv(annotation, sep=" ", header=None)
            df_annotation.columns = ["track", "xmin", "ymin", "xmax", "ymax",
                                     "frame", "lost", "occluded", "generated", "label"]

            for frame in glob.glob(os.path.join(frames_dir, '*.png')):
                frame_name = frame.strip('.png').split('/')[-1]
                frame_area_name, frame_videoi, frame_number = frame_name.split('_')
                frame_number = frame_number.strip('frame')
                if frame_area_name != area_name or videoi != frame_videoi: continue

                image = Image.open(frame)
                width, height = image.size
                relevant_annotation = df_annotation[df_annotation['frame'] == int(frame_number)]
                relevant_annotation = relevant_annotation[relevant_annotation['label'] != 'Cart']
                relevant_annotation = relevant_annotation[(relevant_annotation['occluded'] == 0) &
                                                          (relevant_annotation['lost'] == 0)]

                df_label = relevant_annotation['label'].apply(lambda x: Sdd2Coco[x])

                relevant_annotation = relevant_annotation[["xmax", "xmin", "ymax", "ymin"]]
                # relevant_annotation = relevant_annotation.apply(pd.to_numeric)

                df_width = relevant_annotation['xmax'] - relevant_annotation['xmin']
                df_height = relevant_annotation['ymax'] - relevant_annotation['ymin']
                df_xcenter = df_width.div(2) + relevant_annotation['xmin']
                df_ycenter = df_height.div(2) + relevant_annotation['ymin']

                df_width = df_width.div(width)
                df_height = df_height.div(height)
                df_xcenter = df_xcenter.div(width)
                df_ycenter = df_ycenter.div(height)

                final_annotation = pd.concat([df_label, df_xcenter, df_ycenter, df_width, df_height], axis=1)
                annotation_name = os.path.join(area_output, frame_name)
                final_annotation.to_csv(annotation_name + '.txt', header=False, index=False, sep=' ', mode='w', float_format='%.6f')


def create_random_annoatated(num_to_generate):
    all_frames = []
    for path, subdirs, files in os.walk(path_out):
        for frame in files:
            all_frames.append(os.path.join(path, frame))

    to_generate = random.choices(all_frames, k=num_to_generate)
    for image in to_generate:
        annotation_file = image.replace('frames', 'labels').replace('.png', '.txt')
        im = Image.open(image)
        width, height = im.size

        # Create plot
        plt.figure()
        fig, ax = plt.subplots(1)
        img = np.array(im)
        ax.imshow(img)
        with open(annotation_file, 'r') as f:
            annotations = f.read().strip().split('\n')
            annotations = [x.split(' ') for x in annotations]

        for ann in annotations:
            cat, x_ctr, y_ctr, w, h = ann
            color = bbox_colors[cat]
            # Create a Rectangle patch
            xmin = (float(x_ctr) - float(w)/2) * width
            ymin = (float(y_ctr) - float(h)/2) * height
            bbox = patches.Rectangle((xmin, ymin), width * float(w), height * float(h),
                                     linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            # plt.text(
            #     float(xmin),
            #     float(ymin),
            #     s=cat_names[cat],
            #     color="white",
            #     verticalalignment="top",
            #     bbox={"color": color, "pad": 0},
            # )
        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        sample_name = image.split('/')[-1]
        plt.savefig(os.path.join(samples_out, sample_name), dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close('all')






if __name__ == "__main__":
    # logger_file_fd = open(logger_file, 'a')
    # try:
    #     video_2_image(logger_file_fd)
    # except:
    #     logger_file_fd.close()

    # create_annotations()
    create_random_annoatated(num_to_generate=10)
