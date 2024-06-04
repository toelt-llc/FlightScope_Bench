from glob import glob
from PIL import Image
from ultralytics import YOLO
from utils import draw_bbox
import gradio as gr
import numpy as np
import subprocess


# TODO add counts 
# count = 42

with gr.Blocks() as demo:
    gr.Markdown("Detect planes demo.")

    # models=["yolov8.1-airbus", "yolov8.1-finetuned", "yolov5-base"] # Currently not all working
    models=["SSD", "FasterRCNN", "CenterNet", "RetinaNet", "DETR", "RTMDET", "YOLOv5", "YOLOv8"] 

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                # images_input = gr.File(file_types=["image"],  type="binary", file_count="multiple") #type="binary",
                image_input_single = gr.Image()
            image_output = gr.Image(visible = True)
            # images_output = gr.Gallery(visible = False)
        with gr.Row():
            drop = gr.Dropdown([m for m in models], label="Model selection", type ="index", value=models[0])
            image_button = gr.Button("Detect", variant = 'primary')
            # image_button2 = gr.Button("Count")
            with gr.Column(visible=True) as output_row:
                object_count = gr.Textbox(value = 0,label="Aircrafts detected")

    def runmodel(input_img, model_num):
        Image.fromarray(input_img).save(source:="inptest.jpg")
        print("Using model", model_name:=models[model_num])
        # print(np.shape(input_img)[:2])

        conf = 0.3

        if model_name in models[:-2]:
            cmd = f"python3 image_inference.py {source} inference_test/{model_name.lower()}_config.py --weights {model_name.lower()}best.pth --out-dir inference_test/results/{model_name.lower()}_inference --pred-score-thr {conf}"
            subprocess.run(cmd, shell=True)
            im, count = draw_bbox(model_name.lower())

        if model_name == "YOLOv5":
            cmd = f"python3 yolov5/detect.py --weights yolov5best.pt --source {source}  --save-txt --save-conf --project inference_test/results/yolov5_inference --name predict"
            subprocess.run(cmd, shell=True)
            im, count = draw_bbox(model_name.lower())

        if model_name == "YOLOv8":
            model = YOLO('yolov8best.pt')
            results = model(source, imgsz=1024, conf = conf, save_txt = True, save_conf = True, save = True, project = "inference_test/results/yolov8_inference")
            im, count = draw_bbox(model_name.lower())

            # ADD COUNT
            # try:
            #     with open(f"{exps[-1]}"+"labels/image0.txt", 'r') as fp: lbl_count = len(fp.readlines())
            # except:
            #     lbl_count = 0

        return im, count
    
    # def output(path):
    #     return {output_row: gr.update(visible = True),
    #             object_count: count}
    #             #n_classes: n}
    # image_button2.click(output, inputs=image_input_single, outputs=[object_count, output_row])

    image_button.click(runmodel, inputs=[image_input_single, drop], outputs=[image_output, object_count])
    # examples = gr.Examples(inputs=[Image.open("inptest.jpg")])    

demo.launch()