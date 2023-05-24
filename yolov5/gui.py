from PIL import ImageTk, Image
import cv2
import tkinter as tk
import threading

class VideoPlayer:
    def __init__(self, master):
        self.master = master
        self.master.geometry("1900x1400")
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
        tk.Label(self.master, text="Traffic Sign Detection", font=("Helvetica", 20)).place(x=635, y=20)

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

        # Initialize the video capture and thread
        self.capture = cv2.VideoCapture(0)
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

        self.running = False

        self.thread.start()
        
    def start(self):
        self.running = True
        
    def stop(self):
        self.running = False
        
    def exit(self):
        self.running = False
        self.master.destroy()
        
    def choose_file(self):
        self.running = False
        
    def update(self):
        while True:
            if self.running and self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    frame = cv2.resize(frame, (int(frame.shape[1] * 2), int(frame.shape[0] * 2)))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]
                    if w > 1500 or h > 730:
                        # Calculate the scaling factor for each dimension
                        width_ratio = 1500 / w
                        height_ratio = 730 / h
                        # Use the smaller scaling factor to resize the image
                        scaling_factor = min(width_ratio, height_ratio)
                        w = int(w * scaling_factor)
                        h = int(h * scaling_factor)
                        frame = cv2.resize(frame, (w, h))
                    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                    self.canvas.create_image((1500 - w) // 2, (730 - h) // 2, image=photo, anchor="nw")
                    self.canvas.config(width=w, height=h)
                    self.canvas.image = photo
            else:
                # self.canvas.delete("all")
                photo = ImageTk.PhotoImage(image=Image.open("background.jpg"))
                h, w = photo.height(), photo.width()
                print(h, w)
                self.canvas.create_image(0, 2, image=photo, anchor="nw")
                self.canvas.config(width=w, height=h)
                self.canvas.image = photo

            self.master.update_idletasks()
            self.master.update()
                 
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayer(root)
    root.mainloop()
