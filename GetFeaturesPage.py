from tkinter import Frame, StringVar, Label, Entry, Button, filedialog, messagebox, ttk, Toplevel
import tkinter as tk
import os
import pickle
import numpy as np
from scipy.io import wavfile
from tkinter import ttk
import tensorflow as tf
import pandas as pd
import queue
import threading
import time

class GetFeaturesPage(Frame):
    def __init__(self, parent, controller):
            super().__init__(parent)

            self.grid_columnconfigure(0, weight=1)
            self.grid_columnconfigure(1, weight=1)
            self.grid_columnconfigure(2, weight=1)
            self.grid_columnconfigure(3, weight=1)
    
            # get the audio file path to display in box
            folder_path_label = Label(self, text= "Folder path")
            folder_path_label.grid(row = 1, column=0, sticky= "ew", padx=5, pady=5)
            self.folder_path_var = StringVar()
            folder_path_var_label = Entry(self, textvariable=self.folder_path_var)
            folder_path_var_label.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
            button_7 = Button(self, text="Select", command=lambda: (print("button_7, browse, clicked"), self.select_folder()))
            button_7.grid(row=1, column=3, sticky="ew", padx=5, pady=5)
            
            # get the audio file path to display in box
            model_label = Label(self, text = "Model")
            model_label.grid(row = 2, column=0, sticky="ew", padx=5, pady=5)
            self.model_path_var = StringVar()
            model_path_var_label = Entry(self, textvariable=self.model_path_var)
            model_path_var_label.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
            button_model = Button(self, text = "Select", command=lambda: self.set_model_path())
            button_model.grid(row = 2, column=3, sticky="ew", padx=5, pady=5)

            button_5 = Button(self, text = "Get features", command=lambda: (print("button_5 clicked"), self.button_5_click_get_features()))
            button_5.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

            button_6 = Button(self, text= "Create WAV files", command=lambda: (print("button_6 clicked"), self.create_wav_files()))
            button_6.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

            self.progress_bar = ttk.Progressbar(self, orient="horizontal", mode="determinate")
            self.progress_bar.grid(row=4, columnspan= 4, column=0, sticky="ew", padx=5, pady=5)

    def select_folder(self):
                    self.folder_path = filedialog.askdirectory()
                    self.folder_path = self.folder_path.replace("/", os.path.sep).replace("\\", os.path.sep)

                    self.folder_path_var.set(os.path.basename(self.folder_path))
                    print("Selected folder:", self.folder_path)

    def load_pickle_files_get_features(self):
                    global image_filtered, image_filtered_labels
                    with open(os.path.join(self.folder_path, "Image_filtered.pkl"), "rb") as f:
                        image_filtered = pickle.load(f)
                    with open(os.path.join(self.folder_path, "Image_filtered_labels.pkl"), "rb") as f:
                        image_filtered_labels = pickle.load(f)

    def create_wav_files(self):
                    # Check if the folder path is defined
                    if not hasattr(self, 'folder_path') or not self.folder_path:
                        messagebox.showwarning("Warning", "Please set the folder path first.")
                        return
                    
                    self.load_pickle_files_get_features()

                    output_base_folder = filedialog.askdirectory(title="Select output folder")
                    output_folder = os.path.join(output_base_folder, "audio")
                    os.makedirs(output_folder, exist_ok=True)

                    total_files = len(image_filtered)
                    self.progress_bar["maximum"] = total_files
                    self.progress_bar["value"] = 0

                    for idx, (data, label) in enumerate(zip(image_filtered, image_filtered_labels)):
                        wav_file = os.path.join(output_folder, f"{label}.wav")
                        wavfile.write(wav_file, 44100, data.astype(np.int16))

                        self.progress_bar["value"] = idx + 1
                        self.progress_bar.update()  # Update the progress bar visually
                        time.sleep(0.01)  # This is just for demonstration purposes,

                    print("WAV files created in", output_folder)

    def get_features(self, messages_queue):
                    messages_queue.put("Loading the encoder...")      
                    
                    if self.model_file:
                        print("Selected model file:", self.model_file)
                        # run the progress window popup
                        
                    else:
                        print("No model file selected.")
                        messages_queue.put("Model not found")
                    
                    try:
                        with open(os.path.join(self.folder_path, "Image_filtered.pkl"), "rb") as f:
                            data = pickle.load(f)
                        with open(os.path.join(self.folder_path, "Image_filtered_labels.pkl"), "rb") as f:
                            ID = pickle.load(f)
                    except FileNotFoundError as e:
                        print(f"Error: {e}. Please make sure the selected folder contains the required pickle files.")
                        return
                            
                    print('Loading the encoder...\nIgnore the warnings')
                    
                    # load the encoder
                    encoder = tf.keras.models.load_model(self.model_file)
                    
                    # Update the message to 'Extracting features...'
                    messages_queue.put("Extracting features...")

                    # reshape to required shape
                    X_test = np.repeat(np.array(data)[..., np.newaxis], 3, axis = 3)  

                    # Get the feature vector representations of the images.
                    feature_vectors = encoder.predict(X_test, batch_size=16, verbose=1)

                    # create a dataframe
                    feature_df = pd.DataFrame(feature_vectors)

                    # add the labels
                    feature_df['ID'] = ID

                    # Create a .csv output file
                    feature_df.to_csv(self.folder_path + "/feature_set.csv", index = False)
                    
                    # Indicate that the function is done
                    messages_queue.put("Done")
                

    def show_loading_popup_get_features(self, messages_queue):
                    
                    # Create a new window as a pop-up
                    loading_popup = Toplevel(self)
                    loading_popup.title("")
                    loading_popup.geometry("200x50")

                    # Create a label for the messages
                    message_label = Label(loading_popup, text="", wraplength=200)
                    message_label.place(x=5, y=5)

                    # Update the messages
                    while True:
                        try:
                            message = messages_queue.get_nowait()
                            if message == "Done":
                                break
                            message_label.config(text=message)
                            loading_popup.update()
                        except queue.Empty:
                            # If the queue is empty, wait a moment and try again
                            loading_popup.update()
                            time.sleep(0.1)
                    
                    # Wait for the feature_extraction_thread to finish and close the loading_popup
                    loading_popup.destroy()

    def run_extract_tweeps_with_popup(self):
                    messages_queue = queue.Queue()
                
                    # Run the get_features function in a separate thread
                    feature_extraction_thread = threading.Thread(target=self.get_features, args=(messages_queue,self.model_file))
                    feature_extraction_thread.start()

                    # Run the progress window popup and get the loading_popup and message_label objects
                    self.show_loading_popup_get_features(messages_queue)

                    feature_extraction_thread.join()

    def set_model_path(self):
                    self.model_file = filedialog.askdirectory(title="Select the model file")
                    self.model_path_var.set(os.path.basename(self.model_file))

    def button_5_click_get_features(self):
                    folder_test = self.folder_path_var.get()

                    if not folder_test :
                        messagebox.showwarning("Warning", "You need to set the File Location")
                        return

                    if not isinstance(folder_test, str):
                        messagebox.showwarning("Warning", "You need to set the File Location")
                        return

                    if not folder_test.strip():
                        messagebox.showwarning("Warning", "You need to set the File Location")
                        return
                    
                    self.run_extract_tweeps_with_popup(self.model_file)

                    print("button_5 clicked")