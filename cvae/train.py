import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from dataloader import MainDataLoader

from cvae import CVAE
import numpy as np
import os
import time
import json
from PIL import Image


from logger import get_logger
LOG = get_logger('finesse')

class Trainer(object):

    def __init__(self, 
                config : dict,
                model : nn.Sequential,
                transform : transforms,
                loss_function,
        ):

        self.num_epochs              = config['num_epochs']
        self.batch_size              = config['batch_size']
        self.learning_rate           = config['learning_rate']
        self.weight_decay            = config['weight_decay']
        self.validation_split        = config['validation_split']
        self.device                  = config['device']

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        self.model                   = model
        self.transform               = transform
        self.loss_function           = loss_function
        self.optimizer               = optimizer
        self.learning_rate_scheduler = learning_rate_scheduler

       
        self.data_dir       = config['data_dir']
        self.ann_filename   = config['ann_filename']
        self.image_dim      = config['image_dim'] 
        self.text_dim       = config['text_dim'] 
        self.latent_dim     = config['latent_dim'] 
        self.image_channels = config['image_channels']

        self.load_dataset()


    def load_dataset(self):


        dataset = MainDataLoader(self.data_dir, self.ann_filename, self.transform, self.image_dim, self.image_channels)

        total_samples = len(dataset)
        validation_samples = int(self.validation_split*total_samples)
        train_samples = total_samples - validation_samples

        train_data, validation_data = random_split(dataset, [train_samples, validation_samples])

        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        self.validation_loader = DataLoader(validation_data, shuffle=True, batch_size=self.batch_size)
        
        LOG.debug('Data is loaded')

    def train(self):

        self.iteration_list  = []
        self.train_loss_list = []
        self.val_loss_list   = []
        self.val_accuracy_list   = []

        self.model.to(self.device)
        
        LOG.debug('Starting training')
        count = 0
        for epoch in range(self.num_epochs):
            for data in self.train_loader:
                count += 1

                images, ann = data

                self.model.train()

                images, ann = images.to(self.device), ann.to(self.device)
                
                reconstructed_images, _, _ = self.model(images, ann)
                current_batch_size, _, _, _ = images.size()
                reshaped_reconstructed_images = torch.reshape(reconstructed_images, (current_batch_size, 
                                                                                     self.image_channels, 
                                                                                     self.image_dim, 
                                                                                     self.image_dim))
                
                loss = self.loss_function(reshaped_reconstructed_images, images)
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                if(count % 20 == 0):
                    LOG.debug(f'Epoch: {epoch+1:02d}, Iteration: {count:5d}, Train Loss: {loss.item():.2f}') 
                    self.train_loss_list.append(loss.cpu().data)
                    self.iteration_list.append(count)

            with torch.no_grad():
                for data in self.validation_loader:

                    self.model.eval()
                    images, ann = data

                    images, ann = images.to(self.device), ann.to(self.device)

                    reconstructed_images, _, _ = self.model(images, ann)
                    current_batch_size, _, _, _ = images.size()
                    reshaped_reconstructed_images = torch.reshape(reconstructed_images, (current_batch_size, 
                                                                                         self.image_channels, 
                                                                                         self.image_dim, 
                                                                                         self.image_dim))
                    loss = self.loss_function(reshaped_reconstructed_images, images)

                    # to generate
                    # sample_text = 'blue jacket'
                    # generated_image = self.model.decode(self.model.reparameterize(torch.randn(1, self.model.latent_dim).to(self.device), sample_text), sample_text)
                    # reshaped_generated_image = Image.fromarray(np.array(torch.reshape(generated_image, (self.image_channels, 
                    #                                                                                     self.image_dim, 
                    #                                                                                     self.image_dim)))
                    #                             )
                    # reshaped_generated_image.save('generated_image_sample_{}.jpeg'.format(epoch))       


            self.learning_rate_scheduler.step()

        LOG.debug('Finished Training')

    
    
def main():

    LOG.debug('Logger is working')

    start_time = time.time()

    config_filepath = os.path.join(os.getcwd(), 'cvae/config.json')
    with open(config_filepath, 'r') as f:
        config = json.load(f)

    model = CVAE(config['image_dim'], config['text_dim'], config['latent_dim'], config['image_channels'])

    transform = transforms.Compose([
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(2.8),
                      transforms.RandomGrayscale(0.2)                         
                ]) 

    loss_function = nn.MSELoss(reduction='sum')

    train_obj = Trainer(config, model, transform, loss_function)
    train_obj.train()

    end_time = time.time()

    LOG.debug('Training time: {:.2f} min'.format((end_time - start_time)/60))



if __name__ == '__main__':
    main()




