import os
import numpy as np
import random
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
from datetime import datetime

# Set the path to the directory containing the generated image data
path = 'C:\\Users\\patri\\kv6007\\generated_images'

# Set the list of classes names
classes = ["Emily", "Michael", "Sophia", "Jacob"]
num_classes = len(classes)

# Load the trained model
model = load_model('best_model.keras')

# Create the main window
window = tk.Tk()
window.title("Letter Notification System")

# Create a frame to hold the date picker and button
top_frame = ttk.Frame(window)
top_frame.pack(pady=10)

# Create a date picker
date_label = ttk.Label(top_frame, text="Select Date:")
date_label.pack(side=tk.LEFT, padx=5)
date_picker = DateEntry(top_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
date_picker.pack(side=tk.LEFT, padx=5)

# Create a button to trigger random image selection
select_button = ttk.Button(top_frame, text="Check", command=lambda: select_random_images(date_picker.get_date()))
select_button.pack(side=tk.LEFT, padx=5)

# Create a frame to hold the images
frame = ttk.Frame(window)
frame.pack(pady=10)

# Create a label to display the class counts
count_label = ttk.Label(window, text="")
count_label.pack()

# Function to select random images and display class counts
def select_random_images(selected_date):
    # Clear previous images
    for widget in frame.winfo_children():
        widget.destroy()

    # Select a random number of images between 1 and 10
    num_images = random.randint(1, 10)

    # Initialize class counts
    class_counts = {cls: 0 for cls in classes}

    # Randomly select images and display them
    for _ in range(num_images):
        # Randomly select a class
        class_name = random.choice(classes)

        # Randomly select an image from the class
        class_path = os.path.join(path, class_name)
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        image_name = random.choice(images)
        image_path = os.path.join(class_path, image_name)

        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((128, 128))
        img = np.array(img)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Make predictions
        predictions = model.predict(img)
        predicted_class = classes[np.argmax(predictions)]

        # Update class counts
        class_counts[predicted_class] += 1

        # Display the image
        img = Image.open(image_path)
        img = img.resize((100, 100))
        img = ImageTk.PhotoImage(img)
        label = ttk.Label(frame, image=img)
        label.image = img
        label.pack(side=tk.LEFT, padx=5)

    # Display class counts
    count_text = f"Letters for {selected_date}:\n"
    for cls, count in class_counts.items():
        count_text += f"{cls}: {count}\n"
    count_label.config(text=count_text)

    # Log the class counts with date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{selected_date} {current_time}\n"
    for cls, count in class_counts.items():
        log_entry += f"{cls}: {count}\n"
    log_entry += "\n"

    with open("Letter_Notification.log", "a") as log_file:
        log_file.write(log_entry)

# Start the GUI event loop
window.mainloop()