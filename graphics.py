import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
SIZE = 280
BRUSH_SIZE = 10
import cv2

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mnist Classifier")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(self.root, width=SIZE, height=SIZE, background="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.root.bind("<Delete>", self.clear)
        self.canvas.focus_set()
        self.image = Image.new("L", (SIZE, SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.predict_button = tk.Button(
            self.root,
            text="Predict",
            font=("Segoe UI", 12, "bold"),
            fg="black",
            bg="#b5b7b5",
            activeforeground="white",
            activebackground="#1D4ED8",
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            padx=24,
            pady=8,
            command=self.predict
        )
        self.clear_button = tk.Button(
            self.root,
            text="Clear",
            font=("Segoe UI", 12, "bold"),
            fg="black",
            bg="#b5b7b5",
            activeforeground="white",
            activebackground="#1D4ED8",
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            padx=24,
            pady=8,
            command = self.clear
        )
        self.predict_button.pack(pady=10)
        self.clear_button.pack(pady=10)
    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - BRUSH_SIZE, y - BRUSH_SIZE, x + BRUSH_SIZE, y + BRUSH_SIZE, fill="white", outline="white")
        self.draw.ellipse([x - BRUSH_SIZE, y - BRUSH_SIZE, x + BRUSH_SIZE, y + BRUSH_SIZE], fill="white", outline="white")

    def clear(self, event=None):
        print("Canvas cleared")
        self.canvas.delete("all")

    def predict(self):
        print("Predicting...")
        image = np.array(self.image)
        cv2.imshow("neger", image)