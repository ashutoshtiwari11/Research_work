import tkinter as tk
from PIL import Image, ImageTk
import urllib.request
import cv2
import numpy as np
import os
import csv
import time
from datetime import datetime
from threading import Thread

# Constants
url = 'http://192.168.247.114/cam-hi.jpg'
status_url = 'http://192.168.247.114/status'
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
output_folder = 'output'
csv_file = 'detection_log.csv'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize CSV file with headers if it doesn't exist
if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Class', 'Confidence'])

# Load class names
classesfile = 'coco.names'
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load YOLO model
modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

class SoilStatusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Soil Status")

        # Create label to display status image
        self.status_label = tk.Label(root)
        self.status_label.pack()

        # Create label to display the processed image
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Start the image capture and processing loop in a separate thread
        self.thread = Thread(target=self.process_images)
        self.thread.daemon = True
        self.thread.start()

    def set_status(self, status_code):
        # Load image based on status code
        if status_code == 1:
            image_path = "happy.jpeg"  # Soil is wet, serve happy soil image
        else:
            image_path = "oip.jpeg"  # Soil is dry, serve tinker water me image

        # Open image file
        image = Image.open(image_path)

        # Resize image to fit in the label
        image = image.resize((800, 800), Image.ANTIALIAS)

        # Convert image to Tkinter PhotoImage
        self.status_image = ImageTk.PhotoImage(image)

        # Update label with status image
        self.status_label.configure(image=self.status_image)

    def process_images(self):
        while True:
            try:
                # Read status from URL
                status_req = urllib.request.Request(status_url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urllib.request.urlopen(status_req)
                status_code = int(resp.read().decode())
                self.root.after(0, self.set_status, status_code)

                # Read image from URL
                img_req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                img_resp = urllib.request.urlopen(img_req)
                imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                im = cv2.imdecode(imgnp, -1)

                # Prepare image for YOLO
                blob = cv2.dnn.blobFromImage(im, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                layernames = net.getLayerNames()
                outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(outputNames)

                # Find objects in the image
                im, detected_objects = self.find_objects(outputs, im)

                # Save processed image with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_image_path = os.path.join(output_folder, f'detection_{timestamp}.jpg')
                cv2.imwrite(output_image_path, im)

                # Log detected objects to CSV file
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for obj in detected_objects:
                        writer.writerow([timestamp, obj[0], obj[1]])

                # Convert image to Tkinter PhotoImage
                im_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                im_tk = ImageTk.PhotoImage(im_pil)

                # Update label with the processed image
                self.root.after(0, self.image_label.configure, {'image': im_tk})
                self.image_label.image = im_tk  # Keep a reference to avoid garbage collection

                # Wait for 60 seconds before the next iteration
                time.sleep(60)

            except Exception as e:
                print(f'Error: {e}')
                break

    def find_objects(self, outputs, im):
        hT, wT, cT = im.shape
        bbox = []
        classIds = []
        confs = []
        detected_objects = []

        # Extract bounding boxes and confidences
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))
        
        # Perform non-max suppression
        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        
        # Draw bounding boxes and labels on the image, and record detected objects
        if len(indices) > 0:
            for i in indices.flatten():
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                label = classNames[classIds[i]]
                confidence = int(confs[i] * 100)
                
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(im, f'{label.upper()} {confidence}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Record the detected object
                detected_objects.append((label, confidence))
        
        return im, detected_objects

# Create Tkinter window
root = tk.Tk()
app = SoilStatusApp(root)
root.mainloop()
