import os
path_this = os.path.abspath(os.path.dirname(__file__))

class DataLabels:
    def __init__(self, file_path=os.path.join(path_this, "labels.txt")):
        """Initialize DataLabels with a label file path."""
        self.file_path = file_path
        self.labels_list = self._load_labels()
    
    def _load_labels(self):
        """Load labels from the file, handling errors if the file is missing."""
        try:
            with open(self.file_path, "r") as file:
                return file.read().splitlines()
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            return []
    
    def get_labels(self, index):
        """Return the list of labels."""
        return self.labels_list[index]

    