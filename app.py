import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, Frame, Label, Canvas, Scrollbar
from PIL import Image, ImageTk
import threading
import pyttsx3
import time


class ColorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Detection - Object Tracking")
        self.root.geometry("1600x900")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit("Cannot open webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize Text-to-Speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        # Define color ranges in HSV
        self.color_ranges = {
            'Red': [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                    (np.array([170, 100, 100]), np.array([180, 255, 255]))],
            'Green': [(np.array([35, 100, 100]), np.array([85, 255, 255]))],
            'Blue': [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
            'Yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
            'Orange': [(np.array([10, 100, 100]), np.array([20, 255, 255]))],
            'Purple': [(np.array([130, 100, 100]), np.array([170, 255, 255]))],
            'Pink': [(np.array([140, 50, 100]), np.array([170, 255, 255]))],
            'White': [(np.array([0, 0, 200]), np.array([180, 50, 255]))],
            'Black': [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
            'Gray': [(np.array([0, 0, 50]), np.array([180, 50, 200]))],
        }
        
        # Selected color tracking
        self.selected_color = tk.StringVar(value="Red")
        self.selected_color_start_time = None
        self.selected_color_duration = 0
        self.was_detected = False
        self.announced = False
        self.last_tracking_duration = 0  # Store last tracking time
        
        # Main frame
        main_frame = Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left partition - Original Image
        left_frame = Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        Label(left_frame, text="Original Image", font=("Arial", 12, "bold")).pack()
        self.original_label = Label(left_frame, bg="black")
        self.original_label.pack(fill=tk.BOTH, expand=True)
        
        # Right partition - Detection & Color List
        right_frame = Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        Label(right_frame, text="Tracked Object", font=("Arial", 12, "bold")).pack()
        
        # Color Selection Panel
        color_select_frame = Frame(right_frame)
        color_select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        Label(color_select_frame, text="Select Color:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        color_options = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "White", "Black", "Gray"]
        color_dropdown = ttk.Combobox(color_select_frame, textvariable=self.selected_color, values=color_options, state="readonly", width=15)
        color_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
        color_dropdown.bind("<<ComboboxSelected>>", lambda e: self.reset_tracking())
        
        # Status frame
        status_frame = Frame(right_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Time duration label
        self.time_label = Label(status_frame, text="Duration: 0.0s", font=("Arial", 12, "bold"), fg="darkgreen")
        self.time_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Status label
        self.status_label = Label(status_frame, text="Waiting...", font=("Arial", 10, "bold"), fg="red")
        self.status_label.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Detected image
        self.detected_label = Label(right_frame, bg="black")
        self.detected_label.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Tracked Object Info Section
        info_label = Label(right_frame, text="Tracked Object Info:", font=("Arial", 11, "bold"))
        info_label.pack(anchor="w", padx=10, pady=(5, 2))
        
        canvas = Canvas(right_frame, height=200, bg="white")
        scrollbar = Scrollbar(right_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.info_frame = Frame(canvas, bg="white")
        
        self.info_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.info_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control panel
        control_frame = Frame(root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        Label(control_frame, text="Sensitivity:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.sensitivity_var = tk.IntVar(value=500)
        sensitivity_scale = ttk.Scale(
            control_frame, 
            from_=100, 
            to=5000, 
            variable=self.sensitivity_var, 
            orient=tk.HORIZONTAL
        )
        sensitivity_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.sensitivity_label = Label(control_frame, text="500", font=("Arial", 10))
        self.sensitivity_label.pack(side=tk.LEFT, padx=5)
        
        # Reset tracking button
        reset_btn = ttk.Button(control_frame, text="Reset", command=self.reset_tracking)
        reset_btn.pack(side=tk.RIGHT, padx=5)
        
        # Quit button
        quit_btn = tk.Button(control_frame, text="QUIT", command=self.on_closing, 
                            font=("Arial", 10, "bold"), bg="red", fg="white", padx=10)
        quit_btn.pack(side=tk.RIGHT, padx=10)
        
        self.sensitivity_var.trace("w", self.update_sensitivity_label)
        
        self.running = True
        self.frame_count = 0
        self.update_frame()
    
    def update_sensitivity_label(self, *args):
        self.sensitivity_label.config(text=str(self.sensitivity_var.get()))
    
    def reset_tracking(self):
        """Reset tracking for selected color"""
        self.was_detected = False
        self.announced = False
        self.selected_color_start_time = None
        self.selected_color_duration = 0
        self.last_tracking_duration = 0
        self.time_label.config(text="Duration: 0.0s", fg="darkgray")
        self.status_label.config(text="Ready...")
    
    def announce_color(self, color_name):
        """Pronounce color name in a separate thread"""
        if not self.announced:
            self.announced = True
            def speak():
                try:
                    self.tts_engine.say(f"Detected {color_name}")
                    self.tts_engine.runAndWait()
                except:
                    pass
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
    
    def detect_and_track_best_object(self, frame):
        """Detect the selected color and return the best matching object"""
        selected = self.selected_color.get()
        
        if selected not in self.color_ranges:
            return None, None
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges = self.color_ranges[selected]
        
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, mask
        
        # Find the largest contour (best matching object)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Only track if area is significant
        if area > self.sensitivity_var.get():
            x, y, w, h = cv2.boundingRect(largest_contour)
            return {'contour': largest_contour, 'rect': (x, y, w, h), 'area': area}, mask
        
        return None, mask
    
    def process_frame_with_tracking(self, frame, tracked_obj):
        """Draw tracking box on detected object"""
        detected_frame = frame.copy()
        
        if tracked_obj:
            x, y, w, h = tracked_obj['rect']
            # Draw bright green box
            cv2.rectangle(detected_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Add color name
            color_name = self.selected_color.get()
            cv2.putText(detected_frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Add time on frame
            if self.was_detected:
                cv2.putText(detected_frame, f"Time: {self.selected_color_duration:.1f}s", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        return detected_frame
    
    def update_tracking_info(self, tracked_obj):
        """Update the tracking info panel"""
        for widget in self.info_frame.winfo_children():
            widget.destroy()
        
        if tracked_obj:
            color_name = self.selected_color.get()
            area = tracked_obj['area']
            size_w, size_h = tracked_obj['rect'][2], tracked_obj['rect'][3]
            
            # Display active tracking info
            Label(self.info_frame, text="TIME TRACKING", font=("Arial", 10, "bold")).pack()
            Label(self.info_frame, text=f"{self.selected_color_duration:.2f} sec", 
                  font=("Arial", 28, "bold"), fg="green").pack()
            
            Label(self.info_frame, text=f"Color: {color_name}", font=("Arial", 10)).pack()
            Label(self.info_frame, text=f"Size: {size_w} × {size_h} px", font=("Arial", 10)).pack()
            Label(self.info_frame, text=f"Area: {area:.0f} px²", font=("Arial", 10)).pack()
        else:
            # Display last tracked time when object lost
            Label(self.info_frame, text="LAST TRACKED", font=("Arial", 10, "bold")).pack()
            
            if self.last_tracking_duration > 0:
                Label(self.info_frame, text=f"{self.last_tracking_duration:.2f} sec", 
                      font=("Arial", 28, "bold"), fg="red").pack()
                Label(self.info_frame, text=f"Color: {self.selected_color.get()}", font=("Arial", 10)).pack()
            else:
                Label(self.info_frame, text="Ready to detect", font=("Arial", 11)).pack()
    
    
    def update_frame(self):
        """Update the main display"""
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            
            # Detect and track best object of selected color
            tracked_obj, mask = self.detect_and_track_best_object(frame)
            
            # Update tracking time
            if tracked_obj:
                if not self.was_detected:
                    # Object just appeared
                    self.selected_color_start_time = time.time()
                    self.was_detected = True
                    self.status_label.config(text="✓ Tracking...", fg="green")
                    self.announce_color(self.selected_color.get())
                
                # Update duration
                self.selected_color_duration = time.time() - self.selected_color_start_time
                self.time_label.config(text=f"Duration: {self.selected_color_duration:.1f}s", fg="darkgreen")
            else:
                if self.was_detected:
                    # Object just lost - save the final tracking duration
                    self.last_tracking_duration = self.selected_color_duration
                    self.status_label.config(text="✗ Lost", fg="red")
                self.was_detected = False
                self.selected_color_duration = 0
                self.time_label.config(text="Duration: 0.0s", fg="darkgray")
            
            # Process frame for display
            detected_frame = self.process_frame_with_tracking(frame, tracked_obj)
            
            # Update tracking info panel
            self.update_tracking_info(tracked_obj)
            
            # Convert for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit
            display_height = 350
            scale = display_height / frame_rgb.shape[0]
            display_width = int(frame_rgb.shape[1] * scale)
            
            frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
            detected_resized = cv2.resize(detected_rgb, (display_width, display_height))
            
            # Convert to PIL
            img_original = Image.fromarray(frame_resized)
            img_detected = Image.fromarray(detected_resized)
            
            # Convert to PhotoImage
            photo_original = ImageTk.PhotoImage(img_original)
            photo_detected = ImageTk.PhotoImage(img_detected)
            
            # Update labels
            self.original_label.config(image=photo_original)
            self.original_label.image = photo_original
            
            self.detected_label.config(image=photo_detected)
            self.detected_label.image = photo_detected
        
        if self.running:
            self.root.after(30, self.update_frame)
    
    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
