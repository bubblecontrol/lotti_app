import tkinter as tk
from GetFeaturesPage import GetFeaturesPage
from GetCalls import GetCallsPage

# Main application class
class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("340x550")
        self.title("Lotti Caller")
        
        # Configure the grid of the main window
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create and arrange LabelFrames using grid
        frame_get_calls = tk.LabelFrame(self, text="Get Calls")
        frame_get_features = tk.LabelFrame(self, text="Get Features")
        frame_get_calls.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        frame_get_features.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
       
        # Instantiate and place pages
        get_calls_page = GetCallsPage(frame_get_calls, self)
        get_features_page = GetFeaturesPage(frame_get_features, self)
        get_calls_page.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        get_features_page.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)


# Running the application
if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
