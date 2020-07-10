from __future__ import division
import sys
import os

from model import * 

import random
import time
import datetime
import argparse
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image




class YOLOv3:
    def __init__(self, weights_path, 
                 config_file, 
                 classes, 
                 img_size=416, 
                 nms_thres=0.4, 
                 conf_thres=0.6):
        self.classes = classes
        # Set up model
        self.config_file = config_file
        self.weights_path = weights_path
        self.model = Darknet(config_file, img_size=img_size).to(device)
        self.img_size = img_size
        if weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(device))['model'])
        self.nms_thres = nms_thres
        self.conf_thres = conf_thres
        
    def __repr__(self):
        head = "Yolo Detector:\n"
        cfg = f"\tconfig_file={self.config_file}\nweights_path={self.weights_path}\n"
        args = f"\tnms_thres={self.nms_thres}, conf_thres={self.conf_thres}, img_size={self.img_size}"
        return head+cfg+args
        
    def detect(self, frame, nms_thres=None, conf_thres=None):
        if nms_thres is None:
            nms_thres = self.nms_thres
        if conf_thres is None:
            conf_thres = self.conf_thres
            
        original_shape = frame.shape[:2] # h,w
        # Extract frame as PyTorch tensor
        if type(frame) is np.ndarray:
            frame=Image.fromarray(frame)
        frame = torchvision.transforms.ToTensor()(frame)
        # Pad to square resolution
        frame, _ = pad_to_square(frame, 0)
        
        # Resize
        frame = resize(frame, self.img_size)
        frame = frame.unsqueeze(0).to(device)
        
        # Get detections
        self.model.eval()
        with torch.no_grad():
            detections = self.model(frame)[0]
            detections = non_max_suppression(detections, conf_thres, nms_thres)
            detections = detections[0] # only one image
            if detections is not None and len(detections):
                # Rescale boxes to original image size
                detections[:, :4] = scale_coords(frame.shape[2:], detections[:, :4], original_shape).round()

            # detection - nx6 (x1, y1, x2, y2, conf, cls))
        return detections
        
    def draw(self, detections, img):
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        thickness = round(5e-3 * (img.shape[0] + img.shape[1]) / 2) + 1

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            color = bbox_colors[int(np.where(unique_labels == int(cls))[0])]
            cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.putText(img, self.classes[int(cls)], (x1, y1 - 2), 0, thickness / 3, [225, 255, 255], thickness=thickness, lineType=cv2.LINE_AA)
            
        return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--output_folder", default="output", type=str, help="path to output folder")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device(device)))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    # create output folder
    Path(opt.output_folder).mkdir(parents=True, exist_ok=True)

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        with open(f"{opt.output_folder}/detections.csv", 'a', newline='') as fd:
            writer = csv.writer(fd)
            number_of_detections = 0 if detections is None else len(detections)
            writer.writerow([img_i, path, number_of_detections])

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                with open(f"{opt.output_folder}/detections.csv", 'a', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(list(map(int, [x1, y1, x2, y2, conf, cls_conf, cls_pred])))
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"{opt.output_folder}/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
