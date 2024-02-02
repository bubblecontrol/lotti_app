import os
from os import listdir
from os.path import isfile, join
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pickle
from collections import Counter
from PIL import Image, ImageTk

class FilterCallsPage():
    def __init__(self, parent):
            # Declare `photo` as an instance variable
            self.photo = None
            self.X = None
            self.Y = None
            self.current_image_index = 0   
            self.image_index = None   

    def display_image(self, direction, canvas, canvas_image):
                
                self.canvas = canvas
                self.canvas_image = canvas_image
                
                # Set the image_index 
                if direction == None:
                      self.image_index = None

                print('Image_index:', self.image_index)

                if self.image_index is None:
                    self.directory = filedialog.askdirectory()  # open file explorer to choose directory
                    files = os.listdir(self.directory)  # list all files in the directory
                    self.image_files = [f for f in files if f.endswith(".png") or f.endswith(".jpg")]  # select image files
                    print(len(self.image_files))
                    if not self.image_files:
                        print("No image files found in the selected directory.")
                        return
                    self.current_image_index = 0
                    self.image_index = 0
                else:
                    if not self.image_files:
                        print("No images available to display.")
                        return
                    self.current_image_index = self.image_index % len(self.image_files)  # Ensure index is within bounds

                # change the current_image_index depending the direction
                if direction == "-":
                    self.current_image_index -= 1
                    print('Previous')
                elif direction == "+":
                    self.current_image_index += 1
                    print('Next')

                # ensure that index isn't negative or too high
                if self.current_image_index < 0:
                    self.current_image_index = 0
                elif self.current_image_index >= len(self.image_files):
                    self.current_image_index = len(self.image_files) - 1

                # Update the image_index attribute
                self.image_index = self.current_image_index

                print('Curent image index:', self.current_image_index)

                image_path = os.path.join(self.directory, self.image_files[self.current_image_index])
                image = Image.open(image_path)

                # Get canvas dimensions
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()

                # Calculate the new dimensions while maintaining the aspect ratio
                img_width, img_height = image.size
                width_ratio = canvas_width / float(img_width)
                height_ratio = canvas_height / float(img_height)
                resize_ratio = min(width_ratio, height_ratio)
                new_width = int(img_width * resize_ratio)
                new_height = int(img_height * resize_ratio)

                # Resize the image
                image_resized = image.resize((new_width, new_height), Image.LANCZOS)
                self.photo = ImageTk.PhotoImage(image_resized)

                # Update the canvas image object
                self.canvas.itemconfigure(self.canvas_image, image=self.photo)

    def remove_current_image(self):
                if not self.image_files:
                    print("No images available to remove")
                    return

                # Show confirmation dialog
                response = messagebox.askyesno(title="Delete Image", message="Are you sure you want to delete this image?")
                
                # Proceed with deletion if the response is 'Yes'
                if response:
                    image_path = os.path.join(self.directory, self.image_files[self.current_image_index])
                    os.remove(image_path)  # Delete the image file
                    print(f"Removed image: {self.image_files[self.current_image_index]}")

                    # Update the image_files list and display the next image
                    self.image_files.pop(self.current_image_index)
                    self.display_image(canvas=self.canvas, canvas_image=self.canvas_image, direction = "+")

    def load_pickled_data_filter_calls(self):

                # Get the name of the folder without the "_images" suffix
                folder_name = os.path.basename(self.directory).replace("_images", "")

                # Get the list of files in the folder with the same name as the directory (without "_images")
                files = os.listdir(os.path.join(os.path.dirname(self.directory), folder_name))

                print(files)

                # Load the pickled data if it exists
                for file in files:
                    if "labels" in file:
                        with open(os.path.join(os.path.dirname(self.directory), folder_name, file), "rb") as f:
                            Y = pickle.load(f)
                        print(f"Loaded pickled data for Y: {file}")
                    else:
                        with open(os.path.join(os.path.dirname(self.directory), folder_name, file), "rb") as f:
                            X = pickle.load(f)
                        print(f"Loaded pickled data for X: {file}")

                return X, Y

    def save_to_pickle_filter_calls(self, X, X_name, Y, Y_name):
                '''
                Save all of the spectrograms to a pickle file.
                '''
                # take the last 8 characters from the images folder name and create 'final' folder
                new_folder_path = self.directory[:-6] + 'final'

                os.makedirs(new_folder_path, exist_ok=True)
                
                with open(os.path.join(new_folder_path, X_name + '.pkl'), 'wb') as f:
                        pickle.dump(X, f)

                print('Saved filtered arrays as: ', os.path.join(new_folder_path, X_name))

                with open(os.path.join(new_folder_path, Y_name + '.pkl'), 'wb') as f:
                        pickle.dump(Y, f)

                print('Saved filtered label data as: ', os.path.join(new_folder_path, Y_name))

                return new_folder_path
            
    def display_id_counts(self, Y_calls):
                # Split the ID from the unique number and store only the ID part
                ids_only = [y.split('_')[0] for y in Y_calls]

                # Count the occurrences of each ID in ids_only
                id_counts = Counter(ids_only)

                # Set the starting position for the text
                x_id, y = 225, 650
                self.column_width = max(len(id) for id in id_counts.keys()) * 10 + 30  # Adjust the spacing between the columns

    def create_headers(self, x_id, y):
                self.canvas.create_text(x_id, y, text="ID", anchor="nw", font=("Arial", 10, "bold"))
                self.canvas.create_text(x_id + self.column_width, y, text="Count", anchor="nw", font=("Arial", 10, "bold"))

                self.create_headers(x_id, y)
                y += 15  # Adjust the vertical spacing between the lines

                # Iterate over the ID counts and display them on the canvas in separate columns
                max_rows = 6
                current_row = 0

                for id, count in self.id_counts.items():
                    if current_row == max_rows:  # If more than max_rows, move to a new set of columns
                        x_id += 2 * self.column_width
                        y = 650  # Reset y to the same level as the first headers
                        self.create_headers(x_id, y)
                        y += 15  # Move y to the second row of the new columns
                        current_row = 0  # Reset current_row

                    self.canvas.create_text(x_id, y, text=id, anchor="nw", font=("Arial", 8))
                    self.canvas.create_text(x_id + self.column_width, y, text=count, anchor="nw", font=("Arial", 8))
                    y += 15  # Adjust the vertical spacing between the lines
                    current_row += 1

    def filter_high_quality(self,):
                # load the pickled data
                X, Y = self.load_pickled_data_filter_calls()

                # get a list of all image files in the image folder
                onlyfiles = [f for f in listdir(self.directory) if isfile(join(self.directory, f))]

                # get the call name
                file_name = [os.path.splitext(x)[0] for x in onlyfiles]

                # find the indexes that contain the IDs of the high quality calls we want
                indexes = [i for i, x in enumerate(Y) if x in file_name]

                # filter each element to get only the high quality calls
                Y_calls = np.array([Y[i] for i in indexes])
                X_calls = np.array([X[i] for i in indexes])

                # add a high quality column to data csv 

                # save as pickled data
                new_file_location = self.save_to_pickle_filter_calls(X=X_calls, X_name='Image_filtered', Y=Y_calls, Y_name='Image_filtered_labels')

                # show a message box with the file location
                messagebox.showinfo(title="Filtering Complete", message=f"The filtered data has been saved at {new_file_location}")

                # get a list of all image files in the image folder
                final_names = [f for f in listdir(self.directory) if isfile(join(self.directory, f))]
                
                # get a print out of the IDs and numbers of calls per individual
                self.display_id_counts(final_names) 