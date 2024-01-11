# FlightScope_Bench
A Benchmark of State-of-the-art Object Detection Algorithms For Aircraft Localization

This repository contains code for a Flightscope benchmark paper. A comparative study of aircraft detection from remote sensing images has been conducted. The study compares multiple algorithms, including `Faster RCNN`, `DETR`, `SSD`, `RTMdet`, `RetinaNet`, `CenterNet`, `YOLOv5`, and `YOLOv8`.

## HRPlanesv2 Dataset

The HRPlanesv2 dataset contains 2120 VHR Google Earth images. To further improve experiment results, images of airports from many different regions with various uses (civil/military/joint) selected and labeled. A total of 14,335 aircrafts have been labelled. Each image is stored as a ".jpg" file of size 4800 x 2703 pixels and each label is stored as YOLO ".txt" format. Dataset has been split in three parts as 70% train, %20 validation and test. The aircrafts in the images in the train and validation datasets have a percentage of 80 or more in size. [Link](https://github.com/dilsadunsal/HRPlanesv2-Data-Set)


## Workflows

The study utilizes two popular deep learning frameworks for object detection:

- [mmdetection](https://github.com/open-mmlab/mmdetection)
  - The installation process for mmdetection is detailed in the [_mmdetection_install.ipynb](./_mmdetection_install.ipynb) notebook.

- [detectron2](https://github.com/facebookresearch/detectron2)
  - The setup for detectron2 is explained in [__algorithms_collection.ipynb](./__algorithms_collection.ipynb) .

- [Ultralytics YOLO](https://github.com/ultralytics/)
  - Information about Ultralytics [YOLOv5](https://github.com/ultralytics/yolov5) and [YOLOv8](https://github.com/ultralytics/ultralytics) is available in [__algorithms_collection.ipynb](./__algorithms_collection.ipynb).

## Annotation Conversion

As the HRPlanesv2 dataset is provided with YOLO annotation (txt file with bounding boxes), conversion to JSON COCO annotation is necessary for detectron2 and mmdetection compatibility. The conversion process is detailed in "__data_collection.ipynb" using the [Yolo-to-COCO-format-converter](https://github.com/Taeyoung96/Yolo-to-COCO-format-converter) repository.

## Evaluation Metrics

Some previw of the results:

![Detected Aircrafts](./inference_test/results/centernet_inference/vis/55aa185a-01c8-4668-ae87-1f1d67d15a08.jpg "CenterNet")
![Detected Aircrafts](./inference_test/results/detr_inference/vis/55aa185a-01c8-4668-ae87-1f1d67d15a08.jpg "CenterNet")


## Copyright Notice

The provided code is based on the following open-source libraries:

- [Ultralytics](https://github.com/ultralytics/yolov5)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

This code is free to use for research and non-commercial purposes.

---

Feel free to customize the README further according to your specific needs and preferences.
