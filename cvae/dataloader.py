import torchvision.transforms as transforms
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import json
import numpy as np
from PIL import Image
import os


class MainDataLoader(Dataset):
    def __init__(self, data_dir, ann_filename, transform, img_dim = 128, img_channels = 3):

        self.data_dir = data_dir
        self.image_data_dir = os.path.join(data_dir, 'compressed_images')
        self.image_names = os.listdir(self.image_data_dir)
        self.image_names = [img for img in self.image_names if img.endswith('.jpg')]

        self.image_ids = [os.path.basename(image).split(".jpg")[0] 
                            for image in self.image_names]  # without .jpg extention
        self.transform = transform
        
        # transform variables
        self.img_w = img_dim
        self.img_h = img_dim
        self.img_channels = img_channels
        self.img_mean = 0
        self.img_std = 0
        
        # load annotations
        self.ann_filename = os.path.join(self.data_dir, ann_filename)
        with open(self.ann_filename, 'r') as f:
            self.image_descriptions = json.load(f)  # image_id : description
            
        # load images
        self.images = {} # image_id : image
        for image_name in self.image_names:
            image_path = os.path.join(self.image_data_dir, image_name)
            image = Image.open(image_path).convert('RGB') # remove extra alpha factor
            
            target_size = (self.img_w, self.img_h)  
            image = image.resize(target_size, Image.LANCZOS)  # resize 
            self.images[image_name] = image
        
        self.set_mean_std() # find mean and std of data
        self.preprocess_descriptions()


    def preprocess_descriptions(self):

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoded_descriptions = {} # image_id : description

        for image_id in self.image_descriptions:
            description = self.image_descriptions[image_id]
            description = description['title'] + description['main_category'] + description['sub_category'] + str(description['description'])
            description = ' '.join(description)
            
            text_encoded = tokenizer(description, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            self.encoded_descriptions[image_id] = text_encoded
        
    
    def set_mean_std(self):
        
        num_images = len(self.images)
        mean = np.zeros(3)
        variance = np.zeros(3)
        
        for image in self.images:
            image = np.array(self.images[image])
            curr_mean = np.mean(image, axis=(0, 1))
            curr_var  = np.var(image,  axis=(0, 1))

            mean += curr_mean
            variance += curr_var

        mean /= num_images
        std = np.sqrt(variance / num_images)
        
        self.img_mean, self.img_std = mean, std


    def get_image_description(self, image_id, kind='encoded'):
        if(kind == 'encoded'):
            return self.encoded_descriptions[image_id]
        else:
            return self.image_descriptions[image_id]
    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]
        image = self.images[image_name] 
        
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(self.img_mean, self.img_std)(image) # normalize
        image = self.transform(image)
        
        image_id = self.image_ids[idx]
        ann = self.get_image_description(image_id, 'encoded')['input_ids'] # use only input ids
        
        return image, ann
