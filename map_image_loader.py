import os
from torch.utils.data import Dataset
from PIL import Image
import utils.utility  # Ensure this module is correctly imported
import torch

class ImageMapLoader(Dataset):
    def __init__(self, root_dir, map_dir, transform=None, sample_per_class=-1):
        """
        Args:
            root_dir (string): Directory with all the images organized by class.
            map_dir (string): Directory with all the corresponding map files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            sample_per_class (int, optional): Number of samples per class. 
                If -1, use all samples.
        """
        self.root_dir = root_dir
        self.map_dir = map_dir
        self.transform = transform
        
        # List all class directories
        self.class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_dirs = utils.utility.sort_filenames_by_number(self.class_dirs)
        
        self.map_dirs = [d for d in os.listdir(map_dir) if os.path.isdir(os.path.join(map_dir, d))]
        self.map_dirs = utils.utility.sort_filenames_by_number(self.map_dirs)
        
        self.data = []  # List to hold tuples (file_path, map_path, label_index)
        
        # Assign a unique label index to each class directory
        self.class_to_idx = {class_dir: idx for idx, class_dir in enumerate(self.class_dirs)}
        
        # Iterate over class directories and collect file paths
        for class_dir, map_dir in zip(self.class_dirs, self.map_dirs):
            class_path = os.path.join(root_dir, class_dir)
            map_path = os.path.join(self.map_dir, map_dir)
            
            file_names = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            map_names = [f for f in os.listdir(map_path) if os.path.isfile(os.path.join(map_path, f))]
            
            file_names = utils.utility.sort_filenames_by_number(file_names)
            map_names = utils.utility.sort_filenames_by_number(map_names)
            
            file_paths = [os.path.join(class_path, f) for f in file_names]
            map_paths = [os.path.join(map_path, f) for f in map_names]
            
            if sample_per_class != -1:
                file_paths = file_paths[:sample_per_class]
                
            label_index = self.class_to_idx[class_dir]
            self.data.extend([(file_path, map_path, label_index) for file_path, map_path in zip(file_paths, map_paths)])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path, map_path, label_index = self.data[idx]
        
        # Load image file
        image = Image.open(file_path).convert('RGB')
        map_tensor = torch.load(map_path)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, map_tensor, label_index



class ImageMapsLoader(Dataset):
    def __init__(self, root_dir, map_dir, transform=None, sample_per_class=-1):
        """
        Args:
            root_dir (string): Directory with all the images organized by class.
            map_dir (list or string): Directory or list of directories with all the corresponding map files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            sample_per_class (int, optional): Number of samples per class. 
                If -1, use all samples.
        """
        self.root_dir = root_dir
        self.map_dirs = map_dir if isinstance(map_dir, list) else [map_dir]
        self.transform = transform
        
        # List all class directories
        self.class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_dirs = utils.utility.sort_filenames_by_number(self.class_dirs)
        
        # Sort each map directory list by class
        sorted_map_dirs = []
        for map_dir in self.map_dirs:
            map_class_dirs = [d for d in os.listdir(map_dir) if os.path.isdir(os.path.join(map_dir, d))]
            map_class_dirs = utils.utility.sort_filenames_by_number(map_class_dirs)
            sorted_map_dirs.append(map_class_dirs)
        
        self.data = []  # List to hold tuples (file_path, list_of_map_paths, label_index)
        
        # Assign a unique label index to each class directory
        self.class_to_idx = {class_dir: idx for idx, class_dir in enumerate(self.class_dirs)}
        
        # Iterate over class directories and collect file paths
        for i, class_dir in enumerate(self.class_dirs):
            class_path = os.path.join(root_dir, class_dir)
            file_names = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            file_names = utils.utility.sort_filenames_by_number(file_names)
            file_paths = [os.path.join(class_path, f) for f in file_names]
            
            if sample_per_class != -1:
                file_paths = file_paths[:sample_per_class]
                
            map_paths = []
            for map_dir, sorted_map_class_dirs in zip(self.map_dirs, sorted_map_dirs):
                map_class_dir = sorted_map_class_dirs[i]
                map_path = os.path.join(map_dir, map_class_dir)
                map_names = [f for f in os.listdir(map_path) if os.path.isfile(os.path.join(map_path, f))]
                map_names = utils.utility.sort_filenames_by_number(map_names)
                map_paths.append([os.path.join(map_path, f) for f in map_names])
            
            label_index = self.class_to_idx[class_dir]
            self.data.extend([(file_path, [map_paths[m_idx][j] for m_idx in range(len(map_paths))], label_index) 
                              for j, file_path in enumerate(file_paths)])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path, map_paths, label_index = self.data[idx]
        
        # Load image file
        image = Image.open(file_path).convert('RGB')
        
        # Load all map tensors
        map_tensors = [torch.load(map_path) for map_path in map_paths]
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, map_tensors, label_index
    
class ValidationSetLoader(Dataset):
    def __init__(self, root_dir,transform=None, sample_per_class=-1):
        """
        Args:
            root_dir (string): Directory with all the images organized by class.
            map_dir (string): Directory with all the corresponding map files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            sample_per_class (int, optional): Number of samples per class. 
                If -1, use all samples.
        """
        self.root_dir = root_dir
        self.map_dir = root_dir
        self.transform = transform
        
        # List all class directories
        self.class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_dirs = utils.utility.sort_filenames_by_number(self.class_dirs)
        
        self.map_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(self.map_dir, d))]
        self.map_dirs = utils.utility.sort_filenames_by_number(self.map_dirs)
        
        self.data = []  # List to hold tuples (file_path, map_path, label_index)
        
        # Assign a unique label index to each class directory
        self.class_to_idx = {idx:class_dir for idx, class_dir in enumerate(self.class_dirs)}
        print(self.class_to_idx)
        # Iterate over class directories and collect file paths
        for class_dir, map_dir in zip(self.class_dirs, self.map_dirs):
            class_path = os.path.join(root_dir, class_dir)
            map_path = os.path.join(self.map_dir, map_dir)
            
            file_names = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f[0] == 'i']
            map_names = [f for f in os.listdir(map_path) if os.path.isfile(os.path.join(map_path, f)) and f[0] != 'i']
            

            file_names = utils.utility.sort_filenames_by_number(file_names)
            map_names = utils.utility.sort_filenames_by_number(map_names)
            
            file_paths = [os.path.join(class_path, f) for f in file_names]
            map_paths = [os.path.join(map_path, f) for f in map_names]
            
            if sample_per_class != -1:
                file_paths = file_paths[:sample_per_class]
                
            label_index = class_dir
            self.data.extend([(file_path, map_path, label_index) for file_path, map_path in zip(file_paths, map_paths)])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path, map_path, label_index = self.data[idx]
        
        # Load image file
        image = torch.load(file_path)
        map_tensor = torch.load(map_path)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, map_tensor, label_index
    
class imageLoader(Dataset):
    def __init__(self, root_dir, transform=None, sample_per_class=-1):
        """
        Args:
            root_dir (string): Directory with all the images organized by class.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            sample_per_class (int, optional): Number of samples per class. 
                If -1, use all samples.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # List all class directories
        self.class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_dirs = utils.utility.sort_filenames_by_number(self.class_dirs)
        
        self.data = []  # List to hold tuples (file_path, label_index)
        
        # Assign a unique label index to each class directory
        self.class_to_idx = {class_dir: idx for idx, class_dir in enumerate(self.class_dirs)}
        
        # Iterate over class directories and collect file paths
        for class_dir in self.class_dirs:
            class_path = os.path.join(root_dir, class_dir)
            file_names = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            file_names = utils.utility.sort_filenames_by_number(file_names)
            file_paths = [os.path.join(class_path, f) for f in file_names]
            if sample_per_class != -1:
                file_paths = file_paths[:sample_per_class]
            label_index = self.class_to_idx[class_dir]
            self.data.extend([(file_path, label_index) for file_path in file_paths])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path, label_index = self.data[idx]
        
        # Load image file
        image = Image.open(file_path).convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label_index