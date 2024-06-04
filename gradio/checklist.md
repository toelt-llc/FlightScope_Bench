## Visual interface for inference with the models

### install 
 - install mmdetect manually ( option 1) + fix : https://github.com/open-mmlab/mmdetection/issues/11063
 - copy ssd weights and ssd configs (ssdbest.pth + ssd_config.py)
 - copy image_inference.py
 - install gradio, ultralytics
 - install yolov5, with requirements  
 ___
 - [x] All models working
 - [x] Models outputs standardized 
    - [x] all saved in inference_test/results except for yolov5
 - [x] Results annotations handling 
    - [x] mmdetect models .json bboxes
    - [x] yolo model .txt bboxes
 - [ ] Color ranges for different detection thresholds
    - [ ] Threshold controlling 