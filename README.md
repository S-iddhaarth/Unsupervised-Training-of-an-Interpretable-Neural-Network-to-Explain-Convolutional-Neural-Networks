# Unsupervised-Training-of-an-Interpretable-Neural-Network-to-Explain-Convolutional-Neural-Networks

This project is designed to run on a Linux environment using Python 3.10.8. Below are the setup instructions and steps to execute the code.

## Setup Instructions

1. **Clone the VanillaNet Repository**  
   First, clone the VanillaNet repository from GitHub:
   ```bash
   git clone https://github.com/huawei-noah/VanillaNet
Follow the instructions provided on the VanillaNet GitHub page for setting up the environment and dependencies. Download the vanillanet_5 weights from the same repository.

2. **Update Imports in main.py**  
    After cloning the repository, update the import statements in main.py to correctly reference the location of the VanillaNet folder.

3. **Download ImageNet-1k Dataset**  
    Download the ImageNet-1k dataset from Kaggle:
    ```bash
    pip install kaggle  
      
   kaggle competitions download -c imagenet-object-localization-challenge
   ```
   you have to set up kaggle account and add access token to .kaggle/kaggle.json for this command to work.
   Create a subset of the training dataset with 10 images per class. Add the path to this subset in the configuration file.
4. **Generate Feature Maps**  
Navigate to the map_extractor folder and run the mapGenerator script to produce the feature maps:  
```bash
cd map_extractor
python mapGenerator.py
```
Update the path to the generated feature maps in the configuration file.  

5. **Run the Main File**  
After setting up the feature maps, you can now run the main.py file:
```bash
python main.py
```

## NOTE
Ensure that all paths in the configuration file are updated according to your directory structure.