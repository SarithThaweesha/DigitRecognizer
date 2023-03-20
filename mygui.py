import tkinter as tk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Load the trained model
model = tf.keras.models.load_model('model_CNN2.h5')

# Creating canvas to draw
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400

root = tk.Tk()
root.title('Digit Recognizer')


main_frame = tk.Frame(root)
main_frame.pack()

canvas = tk.Canvas(main_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
canvas.pack(side=tk.LEFT)

# Creating a frame
button_frame = tk.Frame(main_frame)
button_frame.pack(side=tk.TOP, pady=10, padx=10)

prediction_text = tk.StringVar()


def predict_digit():
    global prediction_text

    # Convert the image in canvas to a numpy array
    filename = "canvas_image.ps"
    canvas.postscript(file=filename, colormode='color')
    img = Image.open(filename)
    img = img.convert('RGB').convert('L')
    img = img.resize((28, 28))

    # Invert the image (black background, white digit)
    img = ImageOps.invert(img)

    # Normalizing of the image
    img_array = np.array(img)
    img_array = img_array / 255.0

    # Reshape the image for the model
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    # Update the prediction text
    prediction_text.set('Prediction: ' + str(digit))


# clear function
def clear_canvas():
    canvas.delete('all')
    prediction_text.set('')


#  handle mouse events and draw on the canvas(functions)
def paint(event):
    # Set the line thickness and color
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=20)


# Binding paint function to the canvas
canvas.bind('<B1-Motion>', paint)

predict_button = tk.Button(button_frame, text='Predict', command=predict_digit)
predict_button.pack(side=tk.LEFT, padx=5)

clear_button = tk.Button(button_frame, text='Clear', command=clear_canvas)
clear_button.pack(side=tk.LEFT, padx=5)

prediction_label = tk.Label(button_frame, textvariable=prediction_text)
prediction_label.pack(side=tk.LEFT, padx=5)

root.mainloop()








