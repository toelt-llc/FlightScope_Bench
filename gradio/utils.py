import os, ast
from glob import glob 
from PIL import ImageFont, ImageDraw, Image

def process_txtfile(filename):
    """
    Read txt annotations files (designed for YOLO xywh format)

    Parameters:
        filename(str): path of the txt annotation file.

    Returns:
        segments: list of bboxes in format xmin, ymin, xmax, ymax (as image ratio)
        confs: list of confidences of the bboxes object detection
    """
    segments = []
    confs = []
    with open(filename, 'r') as file:
        for line in file:
            # print(line)
            line = line.strip().split(' ')
            cls = int(line[0])
            conf = line[5]
            x, y, w, h = map(float, line[1:5])
            x_min = x - (w / 2)
            y_min = y - (h / 2)
            x_max = x + (w / 2)
            y_max = y + (h / 2)
            segment = [x_min, y_min, x_max, y_max]
            segments.append(segment)
            confs.append(conf)

    return segments, confs

def process_jsonfile(filename):
    """
    Read json annotations files (designed for mmdetect dict format)

    Parameters:
        filename(str): path of the json annotation file.
    
    Returns:
        segments: bboxes in format xmin, ymin, xmax, ymax (as px coordinates)
        confs: list of confidences of the bboxes object detection
    """
    with open(filename, 'r') as file:
        line = file.readline().strip() 
        dic = ast.literal_eval(line)
        segments = dic['bboxes']
        confs = dic['scores']
        # labels = dic['labels']

    return segments, confs

def lerp_color(color1, color2, t):
    """
    Linearly interpolate between two RGB colors.
    
    Parameters:
        color1 (tuple): RGB tuple of the first color.
        color2 (tuple): RGB tuple of the second color.
        t (float): Interpolation factor between 0 and 1.
        
    Returns:
        tuple: Interpolated RGB color tuple.
    """
    r = int(color1[0] + (color2[0] - color1[0]) * t)
    g = int(color1[1] + (color2[1] - color1[1]) * t)
    b = int(color1[2] + (color2[2] - color1[2]) * t)
    return r, g, b

def generate_color_palette(start_color, end_color, steps):
    """
    Generate an RGB color palette between two colors.
    
    Parameters:
        start_color (tuple): RGB tuple of the starting color.
        end_color (tuple): RGB tuple of the ending color.
        steps (int): Number of steps between the two colors.
        
    Returns:
        list: List of RGB tuples 
    """
    palette = []
    for i in range(steps):
        t = i / (steps - 1)  # interpolation factor
        color = lerp_color(start_color, end_color, t)
        palette.append(color)

    return palette

def draw_bbox(model_name, results_folder="./inference_test/results/", image_path="inptest.jpg"):
    """
    Draw bounding boxes from mmdetect or yolo formats
    """

    # annotations style
    txt_color=(255, 255, 255)
    yellow=(255, 255, 128)
    black = (0, 0, 0)
    steps = 11                 # Step : 5%
    # (255, 0, 0)  # Red
    # (0, 0, 255)    # Blue
    palette = generate_color_palette((255, 0, 0), (0, 0, 255), steps)
    lw = 9
    font = ImageFont.truetype(font="Pillow/Tests/fonts/FreeMono.ttf", size=48)

    im = Image.open(image_path)
    width, height = im.size
    imdraw = ImageDraw.Draw(im)

    exps = sorted(glob(f"inference_test/results/{model_name}_inference/*", recursive = True))
    # print(exps)
    if model_name[:4] == "yolo":
        annot_file = glob(f"{exps[-1]}/labels/" + "*.txt")[0]
        segments, confs = process_txtfile(annot_file)
    else:
        annot_file = glob(f"{exps[1]}/{image_path[:-4]}.json")[0]
        segments, confs = process_jsonfile(annot_file)
    # print("Result bboxes : " + annot_file)    

    for conf, box in  zip(confs, segments): 
        conf_r = round(float(conf), 3) # round conf

        if conf_r >= 0.5: # 0.5 threshold
            bbox_c = palette[1] #
            if conf_r <= 1.0: bbox_c = palette[-1]
            if conf_r < 0.95: bbox_c = palette[-2]
            if conf_r < 0.90: bbox_c = palette[-3]
            if conf_r < 0.85: bbox_c = palette[-4]
            if conf_r < 0.80: bbox_c = palette[-5]
            if conf_r < 0.75: bbox_c = palette[-6]
            if conf_r < 0.70: bbox_c = palette[-7]
            if conf_r < 0.65: bbox_c = palette[-8]
            if conf_r < 0.60: bbox_c = palette[-9]
            if conf_r < 0.55: bbox_c = palette[-10]

            if model_name[:4] == "yolo": 
                box = [box[0]*width, box[1]*height, box[2]*width, box[3]*height]
            imdraw.rectangle(box, width=lw, outline=bbox_c)  # box

            # label
            w, h = font.getbbox(str(conf_r))[2:4] # text w, h
            imdraw.rectangle([box[0], box[1]-h, box[0]+w+1, box[1]+1], width=3, fill = black)  # box
            imdraw.text([box[0], box[1]-h], str(conf_r), fill=yellow, font=font)
            
        im.save(f"{results_folder}{model_name}_inference/clean.jpg")

    # count
    count = len([i for i in confs if float(i) > 0.5])

    return im, count

