import tkinter as tk
from PIL import Image, ImageTk
import urllib.request

class SoilStatusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Soil Status")

        # Create label to display status image
        self.status_label = tk.Label(root)
        self.status_label.pack()

        # Set initial status
        self.set_status(self.get_status_from_url())

    def get_status_from_url(self):
        url = 'http://192.168.247.114/status'
        try:
            resp = urllib.request.urlopen(url)
            status_code = int(resp.read().decode().strip())  # Assuming the server returns '1' or '0' as a string
        except Exception as e:
            print(f"Failed to fetch status from URL: {e}")
            status_code = 0  # Default to dry soil if there's an error

        return status_code

    def set_status(self, status_code):
        # Load image based on status code
        if status_code == 0:
            image_path = "happy.jpeg"  # Soil is wet, serve happy soil image
        else:
            image_path = "oip.jpeg"  # Soil is dry, serve tinker water me image

        # Open image file
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
            return

        # Resize image to fit in the label
        image = image.resize((800, 800), Image.ANTIALIAS)

        # Convert image to Tkinter PhotoImage
        self.status_image = ImageTk.PhotoImage(image)

        # Update label with status image
        self.status_label.configure(image=self.status_image)

# Create Tkinter window
root = tk.Tk()
app = SoilStatusApp(root)
root.mainloop()
