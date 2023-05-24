import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from PIL import ImageTk, Image
import os
import tkinter as tk
import threading
from tkinter import filedialog
import timeit
import pygame

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, TracedModel

class TrafficSignDetection:
    def __init__(self, master):
        self.master = master
        self.master.geometry("950x700")
        self.master.title("Traffic Sign Detection")
        self.master.configure(bg="#f0f0f0")

        # Create the top frame
        self.top_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.top_frame.pack(side="top", fill="both", expand=True, padx=20, pady=20)
        
        # Create the canvas
        self.canvas = tk.Canvas(self.top_frame, bg="#ffffff")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Create the bottom frame
        self.bottom_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.bottom_frame.pack(side="bottom", fill="x")
        
        # Add a line to separate the frames
        separator = tk.Frame(self.master, height=2, bd=1, relief="sunken", bg="#cccccc")
        separator.pack(side="top", fill="x", pady=5)

        # Add labels
        tk.Label(self.master, text="Traffic Sign Detection", font=("Helvetica", 20)).place(x=330, y=10)

        # Change button colors and use icons
        self.start_button = tk.Button(self.bottom_frame, text="Start", command=self.start, bg="#4CAF50", fg="white", relief="raised", borderwidth=3)
        self.start_button.config(width=13, height=2)
        self.start_button.pack(side="left", padx=50, pady=10)

        self.stop_button = tk.Button(self.bottom_frame, text="Stop", command=self.stop, bg="#FFC107", fg="white", relief="raised", borderwidth=3)
        self.stop_button.config(width=13, height=2)
        self.stop_button.pack(side="left", padx=85, pady=10)

        self.choose_button = tk.Button(self.bottom_frame, text="Choose file", command=self.choose_file, bg="#2196F3", fg="white", relief="raised", borderwidth=3)
        self.choose_button.config(width=13, height=2)
        self.choose_button.pack(side="left", padx=50, pady=10)

        self.exit_button = tk.Button(self.bottom_frame, text="Exit", command=self.exit, bg="#F44336", fg="white", relief="raised", borderwidth=3)
        self.exit_button.config(width=13, height=2)
        self.exit_button.pack(side="right", padx=50, pady=10)

    def start(self):
        opt.source = 0
        self.running = True
        self.thread = threading.Thread(target=self.detect, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        # self.thread.join()
       
    def exit(self):
        self.running = False
        self.master.destroy()

    def choose_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            print(file_path)
            self.load_file(file_path)
        
    def load_file(self, file_path):
        _, ext = os.path.splitext(file_path)
        if ext.lower() in [".jpg", ".jpeg", ".png", ".mp4"]:
            opt.source = file_path
            self.running = True
            self.thread = threading.Thread(target=self.detect, args=())
            self.thread.daemon = True
            self.thread.start()
    
    def detect(self):
        if self.running:
            pygame.init()
            start = timeit.default_timer()-3
            label = None
            time_fps = 0
            source = str(opt.source)
            webcam = source.isnumeric()

            # Initialize
            set_logging()
            device = select_device(opt.device)
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(opt.weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(opt.img_size, s=stride)  # check img_size

            if not opt.no_trace:
                model = TracedModel(model, device, opt.img_size)

            if half:
                model.half()  # to FP16

            # Second-stage classifier
            classify = False
            if classify:
                modelc = load_classifier(name='resnet101', n=2)  # initialize
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

            # Set Dataloader
            if webcam:
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            old_img_w = old_img_h = imgsz
            old_img_b = 1

            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Warmup
                if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=opt.augment)[0]

                with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                    pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                # Process detections
                if webcam:  # batch_size >= 1
                    im0 = im0s[0].copy()
                else:
                    im0 = im0s

                # Rescale boxes from img_size to im0 size
                pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(pred[0]):
                    if opt.view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Stream results
                if self.running == False:
                    dataset.stop = True
                    self.canvas.delete("all")
                    break
                stop = timeit.default_timer()
                fps=1/(stop-time_fps)
                time_fps = stop
                if webcam and stop - start >= 3 and label is not None:
                    audio_file = os.path.dirname(__file__) + '\\audios_with_silent\\' + label.split(' ')[0] + '.mp3'
                    # playsound(audio_file)
                    pygame.mixer.music.load(audio_file)
                    start = timeit.default_timer()
                    pygame.mixer.music.play()
                    label = None
                if opt.view_img:
                    frame = im0
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]
                    width_ratio = 900 / w
                    height_ratio = 600 / h
                    # Use the smaller scaling factor to resize the image
                    scaling_factor = min(width_ratio, height_ratio)
                    w = int(w * scaling_factor)
                    h = int(h * scaling_factor)
                    frame = cv2.resize(frame, (w, h))
                    cv2.putText(frame, f"FPS: {int(fps)}",(50,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
                    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                    self.canvas.create_image((900 - w) // 2, (600 - h) // 2, image=photo, anchor="nw")
                    self.canvas.config(width=w, height=h)
                    self.canvas.image = photo

            self.master.update_idletasks()
            self.master.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov7_tiny_12label.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=0, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--no-trace', default=True, action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        root = tk.Tk()
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                app = TrafficSignDetection(root)
                app.master.mainloop()
                strip_optimizer(opt.weights)
        else:
            app = TrafficSignDetection(root)
            app.master.mainloop()
