# FlightScope_Bench
A Benchmark of State-of-the-art Object Detection Algorithms For Aircraft Localization

This repository contains code for a Flightscope benchmark paper. A comparative study of aircraft detection from remote sensing images has been conducted. The study compares multiple algorithms, including `Faster RCNN`, `DETR`, `SSD`, `RTMdet`, `RetinaNet`, `CenterNet`, `YOLOv5`, and `YOLOv8`.

The following Gif is a video inference of Barcelona Airport of the trained algorithms with a detection threshold of 70%. You can find the original video at [ShutterStock](https://www.shutterstock.com/video/clip-1023402088-barcelona-airport-top-view-aircraft-terminal-building).

![OD_previewreduced-ezgif com-optimize](https://github.com/toelt-llc/FlightScope_Bench/assets/54261127/0d3b0fb7-6164-43d1-8e02-99e7d56a47f0)


## HRPlanesv2 Dataset

The HRPlanesv2 dataset contains 2120 VHR Google Earth images. To further improve experiment results, images of airports from many different regions with various uses (civil/military/joint) selected and labeled. A total of 14,335 aircrafts have been labeled. Each image is stored as a ".jpg" file of size 4800 x 2703 pixels, and each label is stored as YOLO ".txt" format. Dataset has been split into three parts as 70% train, 20% validation, and test. The aircrafts in the images in the train and validation datasets have a percentage of 80 or more in size. [Link](https://github.com/dilsadunsal/HRPlanesv2-Data-Set)


## Workflows

The study utilizes two popular deep learning frameworks for object detection:

- [mmdetection](https://github.com/open-mmlab/mmdetection)
  - The installation process for mmdetection is detailed in the [_mmdetection_install.ipynb](./_mmdetection_install.ipynb) notebook.

- [detectron2](https://github.com/facebookresearch/detectron2)
  - The setup for detectron2 is explained in [__algorithms_collection.ipynb](./__algorithms_collection.ipynb) .

- [Ultralytics YOLO](https://github.com/ultralytics/)
  - Information about Ultralytics [YOLOv5](https://github.com/ultralytics/yolov5) and [YOLOv8](https://github.com/ultralytics/ultralytics) is available in [__algorithms_collection.ipynb](./__algorithms_collection.ipynb).

## Algorithms brief description (with usefull blog links)

1. SSD:

[SSD](https://medium.com/axinc-ai/mobilenetssd-a-machine-learning-model-for-fast-object-detection-37352ce6da7d) is a real-time object detection algorithm that predicts bounding boxes and class scores for multiple fixed-size anchor boxes at different scales. It efficiently utilizes convolutional feature maps to achieve fast and accurate detection.

2. Faster RCNN:

[Faster-RCNN](https://blog.paperspace.com/faster-r-cnn-explained-object-detection/#:~:text=Faster%20R%2DCNN%20is%20a%20single%2Dstage%20model%20that%20is,traditional%20algorithms%20like%20Selective%20Search.) is a two-stage object detection framework. It employs a region proposal network (RPN) to generate potential bounding box proposals, combining them with deep feature maps for accurate object detection.

3. CenterNet

[CenterNet](https://towardsdatascience.com/centernet-explained-a7386f368962) is a single-stage object detection approach that focuses on predicting object centers and regressing bounding boxes. It achieves high accuracy through keypoint estimation for precise object localization.

4. RetinaNet

[RetinaNet](https://www.analyticsvidhya.com/blog/2022/09/retinanet-advanced-computer-vision/) is recognized for its focal loss, addressing the class imbalance issue in one-stage detectors. By combining a feature pyramid network with focal loss, RetinaNet excels in detecting objects at various scales with improved accuracy.

5. DETR

[DETR](https://medium.com/visionwizard/detr-b677c7016a47) is a transformer-based object detection model that replaces traditional anchor-based methods with a set-based approach. It utilizes the transformer architecture to comprehend global context and achieve precise object localization.

6. RTMdet

[RTMdet](https://mmyolo.readthedocs.io/en/latest/recommended_topics/algorithm_descriptions/rtmdet_description.html) is an advanced object detection model that leverages a novel framework called Rotate to Maximum (RTM) to improve accuracy compared to traditional Faster R-CNN models. The model is effective in handling objects with varying orientations, resulting in improved detection accuracy. However, its computational complexity may impact performance compared to other state-of-the-art models.

7. YOLOv5:

[YOLOv5](https://sh-tsang.medium.com/brief-review-yolov5-for-object-detection-84cc6c6a0e3a#:~:text=YOLOv5%20uses%20the%20methods%20of,for%20detecting%20different%20scales%20targets) utilizes methods of anchor box refinement, PANet feature pyramid network, and CSPNet for detecting different scale targets. It improves accuracy and efficiency in object detection tasks.

8. YOLOv8:

[YOLOv8](https://arxiv.org/abs/2305.09972) introduces advancements in object detection by refining the architecture, incorporating feature pyramid networks, and optimizing the training pipeline. It enhances accuracy and speed in detecting objects.

## Annotation Conversion

As the HRPlanesv2 dataset is provided with YOLO annotation (txt file with bounding boxes), conversion to JSON COCO annotation is necessary for detectron2 and mmdetection compatibility. The conversion process is detailed in "__data_collection.ipynb" using the [Yolo-to-COCO-format-converter](https://github.com/Taeyoung96/Yolo-to-COCO-format-converter) repository.

## Evaluation Metrics

Some previews of the results:

- Preview of detection results of CenterNet and DETR

<img src="./inference_test/results/centernet_inference/vis/55aa185a-01c8-4668-ae87-1f1d67d15a08.jpg" alt="Detected Aircrafts" title="CenterNet" width="542">
<img src="./inference_test/results/detr_inference/vis/55aa185a-01c8-4668-ae87-1f1d67d15a08.jpg" alt="Detected Aircrafts" title="DETR" width="542">

- Bounding box mean average precision, respectively at IoU=0.5 and IoU=0.75
<img width="542" alt="bbox_mAP50" src="https://github.com/toelt-llc/FlightScope_Bench/assets/54261127/3f0cd2e4-0d9b-4de9-b2b7-8e473b486f4d">
<img width="542" alt="bbox_mAP75" src="https://github.com/toelt-llc/FlightScope_Bench/assets/54261127/a4cfabfb-a635-4f1f-bb78-917885790d68">


## Copyright Notice

The provided code is based on the following open-source libraries:

- [Ultralytics](https://github.com/ultralytics/yolov5)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

This code is free to use for research and non-commercial purposes.

---

Feel free to customize the README further according to your specific needs and preferences.
