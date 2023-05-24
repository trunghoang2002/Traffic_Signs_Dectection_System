import argparse
import os
import sys
from pathlib import Path
import torch

from PIL import ImageTk, Image
import cv2
import tkinter as tk
import threading
from tkinter import filedialog
import timeit
import pygame

FILE = Path(__file__).resolve() # path to this file
ROOT = FILE.parents[0]  # path to YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH (allow import the packages and modules within ROOT)
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_img_size, cv2, non_max_suppression, print_args, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

class TrafficSignDetection:
    def __init__(self, master, **opt):
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

        self.opt = opt
        
    def start(self):
        self.opt['source'] = 0
        self.running = True
        self.thread = threading.Thread(target=self.update, args=(self.opt.values()))
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
            self.opt['source'] = file_path
            self.running = True
            self.thread = threading.Thread(target=self.update, args=(self.opt.values()))
            self.thread.daemon = True
            self.thread.start()

    def update(
                self, 
                weights=ROOT / 'yolov5s.pt',  # model path or triton URL
                source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
                data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                imgsz=(640, 640),  # inference size (height, width)
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.45,  # NMS IOU threshold
                max_det=1000,  # maximum detections per image
                device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                view_img=False,  # show results
                classes=None,  # filter by class: --class 0, or --class 0 2 3
                agnostic_nms=False,  # class-agnostic NMS
                augment=False,  # augmented inference
                visualize=False,  # visualize features
                line_thickness=3,  # bounding box thickness (pixels)
                half=False,  # use FP16 half-precision inference
                dnn=False,  # use OpenCV DNN for ONNX inference
                vid_stride=1,  # video frame-rate stride
    ):
        if self.running:
            pygame.init()
            start = timeit.default_timer()-3
            label = None
            time_fps = 0
            source = str(source)
            webcam = source.isnumeric()

            # Load model
            device = select_device(device)
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            # Dataloader
            bs = 1  # batch_size
            if webcam:
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                bs = len(dataset)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

            # Run inference
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
            dt = (Profile(), Profile(), Profile())
            for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    pred = model(im, augment=augment, visualize=visualize)

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # Process predictions
                if webcam:  # batch_size >= 1
                    im0 = im0s[0].copy()
                else:
                    im0 = im0s.copy()

                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                # Rescale boxes from img_size to im0 size
                pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(pred[0]):
                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

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
                if view_img:
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

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5n_12label.engine', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='show results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    root = tk.Tk()
    opt = parse_opt()
    app = TrafficSignDetection(root, **vars(opt))
    app.master.mainloop()