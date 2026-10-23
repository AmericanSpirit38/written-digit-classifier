import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import torch
from mnist_classifier import MnistClassificationModel
SIZE = 280
BRUSH_SIZE = 10

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = MnistClassificationModel()
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()
class App:
    def __init__(self):
        self.menu()

    def menu(self):
        try:
            self.clear_root()
        except:
            self.root = tk.Tk()
        self.root.configure(background="white")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        self.root.title("Menu")

        self.text_box = tk.Frame(self.root, bg="#026efc", padx=20, pady=20)
        self.text_box.pack(pady=20)

        self.welcome_text = tk.Label(
            self.text_box,
            text="Welcome to Mnist Classifier!",
            font=("Segoe UI", 20, "bold"),
            background="#026efc",
            foreground="white"
        )

        self.welcome_text.pack(pady=20)
        button_frame = tk.Frame(self.root, background="white")
        button_frame.pack(anchor="center")

        self.start_mnist_button = tk.Button(
            button_frame,
            text="Mnist predictor",
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
            command=self.Mnist
        )


        self.start_dataset_maker_button = tk.Button(
            button_frame,
            text="Start Dataset",
            font=("Segoe UI", 12, "bold"),
            fg="black",
            bg="#b5b7b5",
            activeforeground="white",
            activebackground="#1D4ED8",
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            padx=24,
            pady=8
        )



        self.start_mnist_button.pack(side="left", padx=5, pady=5)
        self.start_dataset_maker_button.pack(side="right", padx=5, pady=5)
    def Mnist(self):
        self.clear_root()

        self.root.title("Mnist Classifier")
        self.canvas_menu_frame = tk.Frame(self.root, background="white")
        self.canvas_menu_frame.pack(fill="both", expand=True, pady=5)


        self.menu_button = tk.Button(
            self.canvas_menu_frame,
            text="Menu",
            font=("Segoe UI", 10, "bold"),
            fg="black",
            bg="#b5b7b5",
            activeforeground="white",
            activebackground="#1D4ED8",
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            padx=24,
            pady=8,
            command=self.menu
        )
        self.menu_button.place(relx=0.1, rely=0.1, anchor="center")
        self.canvas = tk.Canvas(self.canvas_menu_frame, width=SIZE, height=SIZE, background="black")
        self.canvas.place(relx=0.45, rely=0.5, anchor="center")
        self.prediction_result_text = tk.Label(
            self.canvas_menu_frame,
            text="Prediction:\nConfidence:",
            font=("Segoe UI", 12, "bold"),
            fg="black",
            bg="white"
        )
        self.prediction_result_text.place(relx=0.85, rely=0.5, anchor="center")
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
            command=self.clear
        )
        self.predict_button.pack(pady=10)
        self.clear_button.pack(pady=10)
    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - BRUSH_SIZE, y - BRUSH_SIZE, x + BRUSH_SIZE, y + BRUSH_SIZE, fill="white", outline="white")
        self.draw.ellipse([x - BRUSH_SIZE, y - BRUSH_SIZE, x + BRUSH_SIZE, y + BRUSH_SIZE], fill="white", outline="white")

    def clear(self, event=None):
        print("Canvas cleared")
        self.canvas.delete("all")
        self.image = Image.new("L", (SIZE, SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        print("Predicting...")

        resized_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)

        image_tensor = (
            torch.tensor(np.array(resized_image), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .div(255.0)
            .to(device)
        )

        with torch.no_grad():
            probabilities = model(image_tensor)

        confidence, predicted = probabilities.max(dim=1)

        digit = predicted.item()
        confidence_percent = confidence.item() * 100


        print(f"Predicted digit: {digit}")
        print(f"Confidence: {confidence_percent:.2f}%")
        result_text = f"Prediction: {digit}\nConfidence: {confidence_percent:.2f}%"
        self.prediction_result_text.config(text=result_text)
        self.prediction_result_text.update()