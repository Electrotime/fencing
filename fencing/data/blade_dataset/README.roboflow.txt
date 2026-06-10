
My First Project - v1 2026-06-08 3:52pm
==============================

This dataset was exported via roboflow.com on June 8, 2026 at 8:46 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 680 images.
Object are annotated in YOLOv11 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fit (black edges))

The following augmentation was applied to create 2 versions of each source image:
* 50% probability of horizontal flip
* Random rotation of between -10 and +10 degrees
* Random brigthness adjustment of between -25 and +25 percent


