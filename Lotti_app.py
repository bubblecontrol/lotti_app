# main.py

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Button, filedialog, Label, StringVar, messagebox, ttk, Toplevel, Frame
from pathlib import Path
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa.display
from scipy.signal import find_peaks
import librosa
import pickle
import pandas as pd 
import noisereduce as nr
from scipy.ndimage import binary_closing
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
from tqdm import tqdm
import shutil
import tensorflow as tf
import time
import queue
import threading
from scipy.io import wavfile
from PIL import Image, ImageTk, ImageSequence
from collections import Counter
import os
import sys

class HomePage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        # Buttons
        button_1 = Button(self, text="Button 1", command=lambda: print("button_1 clicked"), relief="flat")
        button_1.grid(row=1, column=0, sticky='ew')  # Expand to fill the column

        button_2 = Button(self, text="Button 2", command=lambda: (print("button_2 clicked"), controller.show_frame(GetCallsPage)), relief="flat")
        button_2.grid(row=1, column=1, sticky='ew')

        button_3 = Button(self, text="Button 3", command=lambda: (print("button_3 clicked"), controller.show_frame(FilterCallsPage)), relief="flat")
        button_3.grid(row=1, column=2, sticky='ew')

        button_4 = Button(self, text="Button 4", command=lambda: (print("button_4 clicked"), controller.show_frame(GetFeaturesPage)), relief="flat")
        button_4.grid(row=1, column=3, sticky='ew')

        # Configure the grid behavior
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
  
     
class GetCallsPage(Frame):
    def __init__(self, parent, controller):
            Frame.__init__(self, parent)
        
            # define functions
            # Initialize variables
            folder_path = ""

            def open_csv_file(file_path):
                data_file = pd.read_csv(file_path)
                
                return data_file

            # Define the function to set the folder path
            def set_folder_path():
                folder_selected = filedialog.askdirectory()
                if folder_selected:
                    global folder_path
                    folder_path = folder_selected + '/'
                    print(folder_path)
                    audiopath_var.set(folder_path)

            # upload csv and save to new folder
            def upload_and_save_csv():
                file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
                if file_path:
                    print(f"Selected file: {file_path}")
                    file_name = Path(file_path).name

                    # Get the directory containing the script and create a new folder within it
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    new_folder_path = os.path.join(script_dir, file_name.rstrip('.csv') + '_data')
                    os.makedirs(new_folder_path, exist_ok=True)

                    new_file_path = os.path.join(new_folder_path, file_name)
                    shutil.copy(file_path, new_file_path)
                    print(f"File saved to: {new_file_path}")
                    
                    # Update the meta_data_var variable with the file name
                    meta_data_var.set(file_name)
                    data_file_path.set(new_file_path)
                    
                    return new_file_path
                else:
                    return None

            # these are the functions to extract the tweep from a Lotti churr call  
            def get_tweep(call, folder_path, start, call_id):
                                y, sr = librosa.load(folder_path + call, offset = start, duration = 0.6, sr = 48000)

                                ## Noisereduce 
                                y = nr.reduce_noise(y=y, sr=sr, thresh_n_mult_nonstationary = 12,  n_fft = 512)

                                # Compute the Mel spectrogram
                                S = librosa.feature.melspectrogram(y = y, sr=sr, n_fft=512, hop_length=16, n_mels=128, fmin=2000, fmax=11000)

                                # Convert to magnitude spectrogram (dB)
                                S_dB = librosa.amplitude_to_db(S, ref=np.max)

                                # Normalize the array to the range [0, 1]
                                S_dB_norm = librosa.util.normalize(S_dB, axis=0, norm=np.inf)

                                # Find the index of the frequency with the highest magnitude at each time point
                                dominant_indices = np.argmax(S, axis=0)

                                # Find the index of the frequency with the highest amplitude at each time point
                                dominant_indices = np.argmax(S, axis=0)

                                # Create a new spectrogram with only the dominant frequencies set to their original amplitude
                                S_dominant = np.zeros(S.shape)

                                # Create a 2D array of indices for proper indexing
                                time_indices = np.arange(S.shape[1])
                                indices_array = np.stack((dominant_indices, time_indices), axis=-1)

                                S_dominant[tuple(indices_array.T)] = S[tuple(indices_array.T)]

                                # Apply close to the binary mask
                                closed_mask = binary_closing(S_dB, structure=np.ones((3, 3)))

                                # Label the connected pixels in the image
                                labels = label(closed_mask)

                                # Set the minimum size of the regions to keep
                                selected_size = selected_region_size_code.get()

                                if not selected_size:
                                        min_size = 50
                                else:
                                        min_size = int(selected_size)

                                # Iterate over the regions in the labeled image
                                for region in regionprops(labels):
                                    # Get the size of the region
                                    size = region.area

                                    # Remove the region if its size is smaller than the minimum size
                                    if size <= min_size:
                                        minr, minc, maxr, maxc = region.bbox
                                        closed_mask[minr:maxr, minc:maxc] = 0

                                # Apply the cleaned mask to the original spectrogram
                                cleaned_spectrogram = S_dB_norm * closed_mask

                                # Add frequency contours
                                contour_number = selected_contour_code.get()
                                if not contour_number:
                                        num_contours = 10
                                else:
                                        num_contours = int(contour_number)
                                
                                min_contour_level = selected_contour_min_code.get()
                                if not min_contour_level:
                                        min_level = 0.4
                                else:
                                        min_level = min_contour_level

                                contour_levels = np.linspace(S_dB_norm.min() + (S_dB_norm.max() - S_dB_norm.min()) * min_level, S_dB_norm.max(), num_contours)

                                # Create a single array that represents the contour plot
                                contour_array = np.zeros_like(cleaned_spectrogram)

                                for i in range(num_contours - 1):
                                    contour_array[(cleaned_spectrogram >= contour_levels[i]) & (cleaned_spectrogram < contour_levels[i + 1])] = i

                                sums = np.sum(contour_array, axis = 0)

                                peaks, _ = find_peaks(sums) 

                                # ensure that there is no failure by creating a fail safe if no peaks are detected
                                if len(peaks) > 0:
                                        call_start = np.min(peaks[0]-15) # give a little bit of space before the call
                                else:
                                        call_start = 1

                                # get the time of the start
                                start_time = librosa.frames_to_time(call_start, sr = sr, hop_length = 16)

                                end_times = start_time + 0.1

                                end_frames = librosa.time_to_frames(end_times, sr = sr, hop_length = 16)

                                # Select the section of the Mel spectrogram corresponding to x and x + 200ms
                                S_dB_section = contour_array[: , call_start:end_frames]

                                print(np.shape(S_dB_section))
                                
                                return S_dB_section, call_id

            def save_to_pickle_get_calls(meta_data_var_label, X_calls, X_file_name, Y_calls, Y_file_name):
                    # Create the folder if it doesn't exist
                    output_folder = os.path.join(os.getcwd(), meta_data_var_label)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    # Save the pickle files in the created folder
                    with open(os.path.join(output_folder, X_file_name), 'wb') as f:
                        pickle.dump(X_calls, f)

                    with open(os.path.join(output_folder, Y_file_name), 'wb') as f:
                        pickle.dump(Y_calls, f)

                    print(f"Data saved to {output_folder}")

            def save_images(images, filenames, meta_data_var_label):
                    """
                    Save an array of images as PNG files.

                    Args:
                        images (numpy.ndarray): An array of images with shape (num_images, height, width, channels).
                        filenames (list): A list of filenames to use for each image.

                    Raises:
                        ValueError: If the length of `filenames` does not match the number of images in `images`.
                    """
                    if len(images) != len(filenames):
                        raise ValueError("The length of `filenames` must match the number of images in `images`.")
                    
                    # Create the folder if it doesn't exist
                    output_folder = os.path.join(os.getcwd(), f"{meta_data_var_label}_images")
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    for i, img in enumerate(images):
                        plt.figure(figsize=(10, 5))
                        plt.imshow(img, origin='lower', aspect='auto', cmap='gray_r')
                        plt.xlabel('Time (frames)')
                        plt.ylabel('Frequency (Hz)')
                        plt.title(filenames[i])
                        plt.tight_layout()
                        # Save images in the created folder
                        plt.savefig(os.path.join(output_folder, filenames[i]))
                        plt.close()

                    print(f"All images saved to {output_folder}")

            def extract_tweeps(data, folder_path, type, file_name):
                
                progress_style = ttk.Style()
                progress_style.theme_use("clam")
                progress_style.configure("custom.Horizontal.TProgressbar",
                                        background="#D9D9D9",
                                        troughcolor="#708D6D",
                                        barcolor="#D9D9D9",
                                        thickness=45)

                progress_bar = ttk.Progressbar(self,
                                            orient="horizontal",
                                            mode="determinate",
                                            length=120,
                                            style="custom.Horizontal.TProgressbar")
                progress_bar.place(x=626.0,
                                y=720.0,
                                width=150.0,
                                height=45.0)
                
                # create the "Saving images" label for display later
                saving_images_label = Label(self, text="Saving images...", bg="#D9D9D9", fg="black")
                saving_images_label.place(x=626.0 + 20, y=720.0 + 12, width=100, height=20)
                saving_images_label.lower()
                
                # create a "Done" label to display later:
                done_label = Label(self, text="Done", bg="#D9D9D9", fg="black")  # Set the background color to match the progress bar
                done_label.place(x=626.0 + 60, y=720.0 + 12, width=30, height=20)  # Adjust x and y values to align with the progress bar
                done_label.lower()  # Hide the done_label by placing it below other widgets

                # progress bar settings
                total_tasks = len(data)  # Replace this with the actual calculation of total tasks based on user inputs                
                progress_bar["maximum"] = total_tasks
                progress_bar["value"] = 0

                # create objects to store the data
                X_calls = []
                Y_calls = []

                # filter the dataframe to the call type you want (at the moment it is only setup to extract tweeps)
                data = data[data['call_type'] == type] # keep only the specified calls  
                
                # carry out extraction on each row in the filtered dataframe
                for i, row in tqdm(data.iterrows(), total=len(data.index)):
                    file_to_find = os.path.join(folder_path, row['sound.files'])
                    
                    print(folder_path,"-" ,file_to_find)
                    
                    progress_bar["value"] = i + 1
                    progress_bar.update()  # Update the progress bar visually
                    time.sleep(0.01)  # This is just for demonstration purposes,
                    
                    if os.path.exists(file_to_find):    
                                    x, y = get_tweep(row['sound.files'], folder_path, row['start'], row['call'])
                                    # we only want to store the array if it contains information
                                    if x.size > 0:
                                        X_calls.append(x)
                                        Y_calls.append(y) 
                                                                               
                    else:
                        print('Cannot find', folder_path + row['sound.files'])  

                # display the "Saving images..." label
                saving_images_label.lift()
                self.update()                 

                # save the unfiltered array 
                save_to_pickle_get_calls(file_name, X_calls, 'tweep_unfiltered.pkl', Y_calls, 'tweep_unfiltered_labels.pkl')

                print('Saving images...')

                # save pngs to file so you can do the filtering afterwards
                save_images(X_calls, Y_calls, file_name)

                # display the "Done" label
                saving_images_label.lower()
                done_label.lift()
                self.update()
                
            # Define a function to set the data file
            def final_combo():
                global data_file
                data_file = open_csv_file(data_file_path.get())
                print('data_file:\n ', data_file, '\nfolder_path: ', folder_path, '\ncall_type: ', selected_code.get())
                extract_tweeps(data_file, audiopath_var.get(), selected_code.get(), os.path.splitext(os.path.basename(meta_data_var.get()))[0])

            def button_5_click_get_calls():
                selected_call_type.set(option_var.get())

                if selected_call_type.get() == "Please select call type":
                    messagebox.showwarning(title="Warning", message="Please select which call type to target")
                elif selected_call_type.get() in ("Triple", "Bar", "M", "Pip"):
                    messagebox.showwarning(title="Warning", message="This call type is not currently supported")
                else:
                    selected_code.set(call_type_codes[selected_call_type.get()])
                    print(selected_call_type.get(), selected_code.get())

            def button_8_click_get_calls():
                folder_path = audiopath_var.get()
                code = selected_code.get()
                meta_data = meta_data_var.get()
                
                if code != 'Ch':
                    messagebox.showwarning("Warning", "Only Churr calls currently supported")
                    return

                if not folder_path or not code or not meta_data:
                    messagebox.showwarning("Warning", "Please fill in all required fields!")
                    return

                if not isinstance(folder_path, str) or not isinstance(code, str) or not isinstance(meta_data, str):
                    messagebox.showwarning("Warning", "Please fill in all required fields with valid values!")
                    return

                if not folder_path.strip() or not code.strip() or not meta_data.strip():
                    messagebox.showwarning("Warning", "Please fill in all required fields with non-empty values!")
                    return

                print("button_8 clicked")
                final_combo()            

            canvas = Canvas(self,bg = "#FFFFFF",height = 895,width = 1440,bd = 0,highlightthickness = 0,relief = "ridge")
            canvas.place(x = 0, y = 0)

            canvas.create_rectangle(0.0,0.0,1440.0,80.0,fill="#F7F6FB",outline="")

            button_1 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_1 clicked"), controller.show_frame(HomePage)),relief="flat")
            button_1.place(x=0.0,y=0.0,width=528.0,height=80.0)

            button_2 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: print("button_2 clicked"),relief="flat")
            button_2.place(x=528.0,y=0.0,width=292.0,height=80.0)

            button_3 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_3 clicked"), controller.show_frame(FilterCallsPage)),relief="flat")
            button_3.place(x=830.0,y=0.0,width=293.0,height=80.0)

            button_4 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_4 clicked"), controller.show_frame(GetFeaturesPage)),relief="flat")
            button_4.place(x=1134.0,y=1.0,width=306.0,height=79.0)

            button_5 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_5 clicked"),button_5_click_get_calls()),relief="flat")
            button_5.place(x=689.0,y=350.0,width=88.0,height=45.0)

            # Call type dropdown menu
            options = ["Please select call type", "Churr", "Triple", "Pip", "M", "Bar"]

            # the call codes
            call_type_codes = {
                "Churr": "Ch",
                "Triple": "T",
                "Pip": "P",
                "M": "M",
                "Bar": "B"
            }
            # Create and configure the style for the dropdown menu
            style = ttk.Style()
            style.configure("Custom.TMenubutton", background="#708D6D", font=("Inika", 12))

            # define the dropdown menu
            option_var = StringVar(self)
            option_var.set(options[0])  # Set the default value

            # Define the dropdown menu
            dropdown = ttk.OptionMenu(self, option_var, options[0], *options, style="Custom.TMenubutton")
            dropdown.place(x=322.0,y=350.0,width=347.0,height=45.0)

            # Variable to store the selected option when the button is clicked
            selected_call_type = StringVar(self)
            selected_code = StringVar(self)

            button_6 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_6 clicked"), upload_and_save_csv()),relief="flat")
            button_6.place(x=689.0,y=222.0,width=88.0,height=45.0)

            # get the audio file path to display in box
            meta_data_var = StringVar()
            meta_data_var_label = Label(self, textvariable=meta_data_var, bg="#708D6D", font=("Inika", 12))
            meta_data_var_label.place(x=322.0, y=222.0, width=347.0, height=45.0)

            button_7 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_7 clicked"), set_folder_path()),relief="flat")
            button_7.place(x=689.0,y=286.0,width=88.0,height=45.0)

            # get the audio file path to display in box
            audiopath_var = StringVar()
            audiopath_var_label = Label(self, textvariable=audiopath_var, bg="#708D6D", font=("Inika", 12))
            audiopath_var_label.place(x=322.0, y=286.0, width=347.0, height=45.0)

            button_8 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_7 clicked"), button_8_click_get_calls()),relief="flat")

            data_file_path = StringVar()

            button_8.place(x=626.0,y=600.0,width=151.0,height=108.0)
            # Call type dropdown menu
            options_contour_no = [5,10,15,20,25]

            # Define the dropdown menu
            options_contour_no_var = StringVar(self)
            options_contour_no_var.set(options_contour_no[1])  # Set the default value
            selected_contour_code = StringVar(self)  # Variable to store the selected option code

            def update_selected_contour_code(option):
                if option in options_contour_no:
                    selected_contour_code.set(options_contour_no[option])
                else:
                    selected_contour_code.set(10)  # Default to 10 if option is not found
                print(selected_contour_code.get())

            dropdown_cont_no = ttk.OptionMenu(self,options_contour_no_var,options_contour_no[1],*options_contour_no,style="Custom.TMenubutton",command=update_selected_contour_code)
            dropdown_cont_no.place(x=322.0,y=537.0,width=120.0,height=45.0)

            # Call type dropdown menu
            options_contour_min = [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

            # Define the contour minimum level menu
            options_contour_min_var = StringVar(self)
            options_contour_min_var.set(options_contour_min[3])  # Set the default value
            selected_contour_min_code = StringVar(self)

            def update_selected_contour_min(option):
                if option in options_contour_no:
                    selected_contour_min_code.set(options_contour_min[option])
                else:
                    selected_contour_min_code.set(0.4)  # Default to 0.4 if option is not found

            dropdown_cont_min = ttk.OptionMenu(self,options_contour_min_var,options_contour_min[3],*options_contour_min,style="Custom.TMenubutton",command=update_selected_contour_min)
            dropdown_cont_min.place(x=322.0,y=600.0,width=120.0,height=45.0)

            # Region size dropdown menu
            options_region_size = [25, 50, 75, 100, 125, 150]

            # Define the region size menu
            options_region_size_var = StringVar(self)
            options_region_size_var.set(options_region_size[1])  # Set the default value
            selected_region_size_code = StringVar(self)

            def update_selected_region_size(option):
                if option in options_region_size:
                    selected_region_size_code.set(option)
                else:
                    selected_region_size_code.set(50)  # Default to 100 if option is not found
                print('Region size = ' + selected_region_size_code.get())
            
            dropdown_region_size = ttk.OptionMenu(self, options_region_size_var, options_region_size[1], *options_region_size, style="Custom.TMenubutton", command=update_selected_region_size)
            dropdown_region_size.place(x=322.0,y=663.0,width=120.0,height=45.0)
                      
            canvas.create_rectangle(322.0,350.0,669.0,395.0,fill="#708D6D",outline="")
            canvas.create_rectangle(322.0,286.0,69.0,331.0,fill="#708D6D",outline="")
            canvas.create_rectangle(322.0,222.0,669.0,267.0,fill="#708D6D",outline="")
            canvas.create_rectangle(322.0,537.0,15.0,582.0,fill="#708D6D",outline="")
            canvas.create_rectangle(322.0,600.0,415.0,645.0,fill="#708D6D",outline="")
            canvas.create_rectangle(322.0,663.0,415.0,708.0,fill="#708D6D",outline="")

            canvas.create_text(83.0,139.0,anchor="nw",
                text="First you need to upload the metadata file, select the call type you are targeting, \nand select the folder where the audio files are stored.",
                fill="#FFFFFF",font=("Inika", 20 * -1)
            )

            canvas.create_rectangle(83.0,350.0,302.0,395.0,fill="#A6B0A5", outline="")
            canvas.create_rectangle(83.0,286.0,302.0,331.0,fill="#A6B0A5",outline="")
            canvas.create_rectangle(83.0,222.0,302.0,267.0,fill="#A6B0A5",outline="")
            
            canvas.create_text(115.0,358.0,anchor="nw",
                text="Target call type",
                fill="#000000",font=("Inika", 20 * -1))

            canvas.create_text(115.0,294.0,anchor="nw",
                text="Audio location",
                fill="#000000",ont=("Inika", 20 * -1)
            )

            canvas.create_text(116.0,232.0,anchor="nw",
                text="Metadata file",
                fill="#000000",font=("Inika", 20 * -1)
            )

            canvas.create_rectangle(83.0,537.0,302.0,582.0,fill="#A6B0A5",outline="")
            canvas.create_rectangle(83.0,600.0,302.0,645.0,fill="#A6B0A5",outline="")
            canvas.create_rectangle(83.0,663.0,302.0,708.0,fill="#A6B0A5",outline="")

            canvas.create_text(108.0,611.0,anchor="nw",
                text="Contour min level",
                fill="#000000",font=("Inika", 20 * -1))

            canvas.create_text(106.0,674.0,anchor="nw",
                text="Region size",
                fill="#000000",font=("Inika", 20 * -1))

            canvas.create_text(115.0,545.0,anchor="nw",
                text="Contour number",
                fill="#000000",font=("Inika", 20 * -1)
            )

            canvas.create_text(87.0,474.0,anchor="nw",
                text="Advanced settings",
                fill="#FFFFFF",font=("Inika", 32 * -1)
            )

class FilterCallsPage(Frame):
    def __init__(self, parent, controller):
            Frame.__init__(self, parent)

            # Declare `photo` as a global variable
            photo = None
            X = None
            Y = None

            def display_image(canvas, canvas_image, direction):
                global photo, image_files, current_image_index, directory, image_index  # Update global variables

                if direction == None:
                    image_index = None
                elif direction == "+" and image_index == None: 
                    current_image_index = 0 + 1
                    image_index = current_image_index
                elif direction == "+" and image_index != None: 
                    current_image_index = image_index + 1
                    image_index = current_image_index
                elif direction == "-" and image_index == None:
                    current_image_index = len(image_files) - 1
                    image_index = current_image_index
                elif direction == "-" and image_index != None: 
                    current_image_index = image_index - 1
                    image_index = current_image_index 

                if image_index is None:
                    directory = filedialog.askdirectory()  # open file explorer to choose directory
                    files = os.listdir(directory)  # list all files in the directory
                    image_files = [f for f in files if f.endswith(".png") or f.endswith(".jpg")]  # select image files
                    if not image_files:
                        print("No image files found in the selected directory.")
                        return
                    current_image_index = 0
                else:
                    if not image_files:
                        print("No images available to display.")
                        return
                    current_image_index = image_index % len(image_files)  # Ensure index is within bounds

                image_path = os.path.join(directory, image_files[current_image_index])  # get the full path of the image file
                image = Image.open(image_path)  # open the image using PIL
                
                # Define the desired canvas size
                canvas_width = 775
                canvas_height = 390

                # Calculate the new dimensions while maintaining the aspect ratio
                img_width, img_height = image.size
                width_ratio = float(canvas_width) / float(img_width)
                height_ratio = float(canvas_height) / float(img_height)
                resize_ratio = min(width_ratio, height_ratio)
                new_width = int(img_width * resize_ratio)
                new_height = int(img_height * resize_ratio)

                # Resize the image
                image_resized = image.resize((new_width, new_height), resample=Image.LANCZOS)
                photo = ImageTk.PhotoImage(image_resized)  # create a PhotoImage object

                canvas.itemconfigure(canvas_image, image=photo)  # replace the current image with the new one
                canvas.itemconfigure(canvas_image, anchor="center") 

            def remove_current_image():
                global image_files, current_image_index, directory

                if not image_files:
                    print("No images available to remove")
                    return

                # Show confirmation dialog
                response = messagebox.askyesno(title="Delete Image", message="Are you sure you want to delete this image?")
                
                # Proceed with deletion if the response is 'Yes'
                if response:
                    image_path = os.path.join(directory, image_files[current_image_index])
                    os.remove(image_path)  # Delete the image file
                    print(f"Removed image: {image_files[current_image_index]}")

                    # Update the image_files list and display the next image
                    image_files.pop(current_image_index)
                    display_image(canvas, canvas_image, direction = "+")

            def load_pickled_data_filter_calls():
                global directory

                # Get the name of the folder without the "_images" suffix
                folder_name = os.path.basename(directory).replace("_images", "")

                # Get the list of files in the folder with the same name as the directory (without "_images")
                files = os.listdir(os.path.join(os.path.dirname(directory), folder_name))

                print(files)

                # Load the pickled data if it exists
                for file in files:
                    if "labels" in file:
                        with open(os.path.join(os.path.dirname(directory), folder_name, file), "rb") as f:
                            Y = pickle.load(f)
                        print(f"Loaded pickled data for Y: {file}")
                    else:
                        with open(os.path.join(os.path.dirname(directory), folder_name, file), "rb") as f:
                            X = pickle.load(f)
                        print(f"Loaded pickled data for X: {file}")

                return X, Y

            def save_to_pickle_filter_calls(X, X_name, Y, Y_name):
                '''
                Save all of the spectrograms to a pickle file.
                '''
                # take the last 8 characters from the images folder name and create 'final' folder
                new_folder_path = directory[:-6] + 'final'

                os.makedirs(new_folder_path, exist_ok=True)
                
                with open(os.path.join(new_folder_path, X_name + '.pkl'), 'wb') as f:
                        pickle.dump(X, f)

                print('Saved filtered arrays as: ', os.path.join(new_folder_path, X_name))

                with open(os.path.join(new_folder_path, Y_name + '.pkl'), 'wb') as f:
                        pickle.dump(Y, f)

                print('Saved filtered label data as: ', os.path.join(new_folder_path, Y_name))

                return new_folder_path
            
            def display_id_counts(Y_calls):
                # Split the ID from the unique number and store only the ID part
                ids_only = [y.split('_')[0] for y in Y_calls]

                # Count the occurrences of each ID in ids_only
                id_counts = Counter(ids_only)

                # Set the starting position for the text
                x_id, y = 225, 650
                column_width = max(len(id) for id in id_counts.keys()) * 10 + 30  # Adjust the spacing between the columns

                def create_headers(x_id, y):
                    canvas.create_text(x_id, y, text="ID", anchor="nw", font=("Arial", 10, "bold"))
                    canvas.create_text(x_id + column_width, y, text="Count", anchor="nw", font=("Arial", 10, "bold"))

                create_headers(x_id, y)
                y += 15  # Adjust the vertical spacing between the lines

                # Iterate over the ID counts and display them on the canvas in separate columns
                max_rows = 6
                current_row = 0

                for id, count in id_counts.items():
                    if current_row == max_rows:  # If more than max_rows, move to a new set of columns
                        x_id += 2 * column_width
                        y = 650  # Reset y to the same level as the first headers
                        create_headers(x_id, y)
                        y += 15  # Move y to the second row of the new columns
                        current_row = 0  # Reset current_row

                    canvas.create_text(x_id, y, text=id, anchor="nw", font=("Arial", 8))
                    canvas.create_text(x_id + column_width, y, text=count, anchor="nw", font=("Arial", 8))
                    y += 15  # Adjust the vertical spacing between the lines
                    current_row += 1

            def filter_high_quality():
                # load the pickled data
                X, Y = load_pickled_data_filter_calls()

                # get a list of all image files in the image folder
                onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

                # get the call name
                file_name = [os.path.splitext(x)[0] for x in onlyfiles]

                # find the indexes that contain the IDs of the high quality calls we want
                indexes = [i for i, x in enumerate(Y) if x in file_name]

                # filter each element to get only the high quality calls
                Y_calls = np.array([Y[i] for i in indexes])
                X_calls = np.array([X[i] for i in indexes])

                # add a high quality column to data csv 

                # save as pickled data
                new_file_location = save_to_pickle_filter_calls(X=X_calls, X_name='Image_filtered', Y=Y_calls, Y_name='Image_filtered_labels')

                # show a message box with the file location
                messagebox.showinfo(title="Filtering Complete", message=f"The filtered data has been saved at {new_file_location}")

                # get a list of all image files in the image folder
                final_names = [f for f in listdir(directory) if isfile(join(directory, f))]
                
                # get a print out of the IDs and numbers of calls per individual
                display_id_counts(final_names)
                    
            canvas = Canvas(self,bg = "#FFFFFF",height = 895,width = 1440,bd = 0,highlightthickness = 0,relief = "ridge")
            canvas.place(x = 0, y = 0)
            canvas.create_rectangle(0.0,0.0,1440.0,80.0,fill="#F7F6FB",outline="")

            button_1 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_1 clicked"), controller.show_frame(HomePage)),relief="flat")
            button_1.place(x=0.0,y=0.0,width=528.0,height=80.0)

            button_2 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_2 clicked"), controller.show_frame(GetCallsPage)),relief="flat")
            button_2.place(x=528.0,y=0.0,width=292.0,height=80.0)

            button_3 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_3 clicked")),relief="flat")
            button_3.place(x=830.0,y=0.0,width=293.0,height=80.0)

            button_4 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_4 clicked"), controller.show_frame(GetFeaturesPage)),relief="flat")
            button_4.place(x=1134.0,y=1.0,width=306.0,height=79.0)

            button_5 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_5, remove, clicked"),remove_current_image()),relief="flat")
            button_5.place(x=429.0,y=585.0,width=115.0,height=45.0)

            button_6 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_6, Done, clicked"), filter_high_quality()),relief="flat")
            button_6.place(x=682.0,y=585.0,width=81.0,height=45.0)

            button_8 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_8, previous, clicked"), display_image(canvas, canvas_image, direction = "-")),relief="flat")
            button_8.place(x=377.0,y=585.0,width=43.0,height=45.0)

            button_9 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_9, Next, clicked"),display_image(canvas, canvas_image, direction = "+")),relief="flat")
            button_9.place(x=553.0,y=585.0,width=43.0,height=45.0)

            canvas.create_rectangle(100.0,167.0,875.0,557.0,fill="#708D6D",outline="")
            canvas.create_rectangle(211.0,643.0,764.0,798.0,fill="#708D6D",outline="")

            # calculate the center point of the rectangle
            center_x = (211.0 + 763.0) / 2
            center_y = (152.0 + 572.0) / 2

            # create the canvas image object
            canvas_image = canvas.create_image(center_x,center_y,anchor="center",image=None)
            button_10 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print('clicked button 10'),display_image(canvas, canvas_image, direction = None)), # set the command to the display_image function
                relief="flat")
            button_10.place(x=211.0,y=585.0,width=80.0,height=45.0)    

class GetFeaturesPage(Frame):
    def __init__(self, parent, controller):
            Frame.__init__(self, parent)
            

            def select_folder():
                    global folder_path
                    folder_path = filedialog.askdirectory()
                    folder_path = folder_path.replace("/", os.path.sep).replace("\\", os.path.sep)

                    folder_path_var.set(os.path.basename(folder_path))
                    print("Selected folder:", folder_path)

            def load_pickle_files_get_features():
                    global image_filtered, image_filtered_labels
                    with open(os.path.join(folder_path, "Image_filtered.pkl"), "rb") as f:
                        image_filtered = pickle.load(f)
                    with open(os.path.join(folder_path, "Image_filtered_labels.pkl"), "rb") as f:
                        image_filtered_labels = pickle.load(f)

            def create_wav_files():
                    load_pickle_files_get_features()

                    output_base_folder = filedialog.askdirectory(title="Select output folder")
                    output_folder = os.path.join(output_base_folder, "audio")
                    os.makedirs(output_folder, exist_ok=True)

                    total_files = len(image_filtered)
                    progress_bar["maximum"] = total_files
                    progress_bar["value"] = 0

                    for idx, (data, label) in enumerate(zip(image_filtered, image_filtered_labels)):
                        wav_file = os.path.join(output_folder, f"{label}.wav")
                        wavfile.write(wav_file, 44100, data.astype(np.int16))

                        progress_bar["value"] = idx + 1
                        progress_bar.update()  # Update the progress bar visually
                        time.sleep(0.01)  # This is just for demonstration purposes,

                    print("WAV files created in", output_folder)

            def get_features(messages_queue, model_file):
                    messages_queue.put("Loading the encoder...")      
                    
                    if model_file:
                        print("Selected model file:", model_file)
                        # run the progress window popup
                        
                    else:
                        print("No model file selected.")
                        messages_queue.put("Model not found")
                    
                    try:
                        with open(os.path.join(folder_path, "Image_filtered.pkl"), "rb") as f:
                            data = pickle.load(f)
                        with open(os.path.join(folder_path, "Image_filtered_labels.pkl"), "rb") as f:
                            ID = pickle.load(f)
                    except FileNotFoundError as e:
                        print(f"Error: {e}. Please make sure the selected folder contains the required pickle files.")
                        return
                            
                    print('Loading the encoder...\nIgnore the warnings')
                    
                    # load the encoder
                    encoder = tf.keras.models.load_model(model_file)
                    
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
                    feature_df.to_csv(folder_path + "/feature_set.csv", index = False)
                    
                    # Indicate that the function is done
                    messages_queue.put("Done")
                

            def show_loading_popup_get_features(messages_queue):
                    
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

            def run_extract_tweeps_with_popup(model_file):
                    messages_queue = queue.Queue()
                
                    # Run the get_features function in a separate thread
                    feature_extraction_thread = threading.Thread(target=get_features, args=(messages_queue,model_file))
                    feature_extraction_thread.start()

                    # Run the progress window popup and get the loading_popup and message_label objects
                    show_loading_popup_get_features(messages_queue)

                    feature_extraction_thread.join()
            
            def button_5_click_get_features():
                    folder_test = folder_path_var.get()

                    if not folder_test :
                        messagebox.showwarning("Warning", "You need to set the File Location")
                        return

                    if not isinstance(folder_test, str):
                        messagebox.showwarning("Warning", "You need to set the File Location")
                        return

                    if not folder_test.strip():
                        messagebox.showwarning("Warning", "You need to set the File Location")
                        return
                    
                    model_file = filedialog.askdirectory(title="Select the model file")
                    model_path_var.set(os.path.basename(model_file))

                    run_extract_tweeps_with_popup(model_file)

                    print("button_5 clicked")
            
            canvas = Canvas(self,bg = "#FFFFFF",height = 895,width = 1440,bd = 0,highlightthickness = 0,relief = "ridge")
            canvas.place(x = 0, y = 0)
            canvas.create_rectangle(0.0,0.0,1440.0,80.0,fill="#F7F6FB",outline="")

            button_1 = Button(self,borderwidth=0, highlightthickness=0, command=lambda: (print("button_1 clicked"), controller.show_frame(HomePage)),relief="flat")
            button_1.place(x=0.0,y=0.0,width=528.0,height=80.0)

            button_2 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_2 clicked"),controller.show_frame(GetCallsPage)),relief="flat")
            button_2.place(x=528.0,y=0.0,width=292.0,height=80.0)

            button_3 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_3 clicked"), controller.show_frame(FilterCallsPage)),relief="flat")
            button_3.place(x=830.0,y=0.0,width=293.0,height=80.0)

            button_4 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_4 clicked")),relief="flat")
            button_4.place(x=1134.0,y=1.0,width=306.0,height=79.0)

            canvas.create_rectangle(327.0,240.0,570.0,285.0,fill="#708D6D",outline="")
            canvas.create_rectangle(327.0,177.0,70.0,222.0,fill="#708D6D",outline="")
            canvas.create_rectangle(27.0,303.0,570.0,348.0,fill="#708D6D",outline="")

            # get the audio file path to display in box
            model_path_var = StringVar()
            model_path_var_label = Label(self, textvariable=model_path_var, fill ="#708D6D", font=("Inika", 12))
            model_path_var_label.place(x=345.0, y=303.0, width=200.0, height=45.0)

            button_5 = Button(self,borderwidth=0, highlightthickness=0,command=lambda: (print("button_5 clicked"), button_5_click_get_features()),relief="flat")
            button_5.place(x=593.0,y=303.0,width=100.0,height=45.0)

            button_6 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_6 clicked"),create_wav_files()),relief="flat")
            button_6.place(x=593.0,y=240.0,width=100.0,height=45.0)

            progress_style = ttk.Style()
            progress_style.theme_use("clam")
            progress_style.configure("custom.Horizontal.TProgressbar",background="#D9D9D9",troughcolor="#708D6D",barcolor = "#D9D9D9",thickness=45)  # Adjust the thickness to match your desired height

            progress_bar = ttk.Progressbar(self,orient="horizontal", mode="determinate", length=242, style="custom.Horizontal.TProgressbar")
            progress_bar.place(x=327, y=240, width= 242, height= 45.0)  
            
            button_7 = Button(self,borderwidth=0,highlightthickness=0,command=lambda: (print("button_7, browse, clicked"), select_folder()),relief="flat")
            button_7.place(x=593.0,y=177.0,width=100.0,height=45.0)

            canvas.create_rectangle(83.0, 177.0,310.0,222.0,fill="#D9D9D9",outline="")

            # get the audio file path to display in box
            folder_path_var = StringVar()
            folder_path_var_label = Label(self, textvariable=folder_path_var, bg="#708D6D", font=("Inika", 12))
            folder_path_var_label.place(x=345.0,  y=177.0, width=200.0, height=45.0)

            canvas.create_rectangle(83.0,303.0,310.0,348.0,fill="#D9D9D9",outline="")
            canvas.create_rectangle(83.0, 240.0, 310.0,285.0,fill="#D9D9D9",outline="")

            canvas.create_text( 83.0,177.0,anchor="nw",text="File location",fill="#000000", font=("Inika", 20 * -1))
            canvas.create_text(83.0,  240.0, anchor="nw",text="Create .wav files",fill="#000000",font=("Inika", 20 * -1))
            canvas.create_text(83.0,303.0,anchor="nw",text="Autocoder features",fill="#000000",font=("Inika", 20 * -1) )

class MainApplication(Tk):
    def __init__(self):
        Tk.__init__(self)

        self.geometry("1428x895")
        self.configure(bg = "#FFFFFF")
       
        self.container = Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (HomePage, GetCallsPage, FilterCallsPage, GetFeaturesPage):
            page_name = F.__name__
            self.frames[page_name] = F

        self.show_frame(HomePage)

    def show_frame(self, page_class):
        page_name = page_class.__name__
        if isinstance(self.frames[page_name], type):
            frame = self.frames[page_name](parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        else:
            frame = self.frames[page_name]
        frame.tkraise()
        frame.lift()  # Add this line to bring the frame to the top

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()