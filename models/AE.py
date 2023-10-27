import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import os
import torchvision
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')
from architectures import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder
from dataset_crafter import AutoEncoderDataset

from PIL import Image
import numpy as np

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        output_size = args.embedding_size
        self.encoder = CNN_Encoder(output_size)

        self.decoder = CNN_Decoder(args.embedding_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, 4096))
        return self.decode(z)

class AE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()
        self.train_loader = self.train_data
        self.test_loader = self.test_data
        self.val_loader = self.val_data

        self.model = Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.best_loss = float('inf')
        self.results_path = r"C:\\Users\\EmirKISA\\Desktop\\Projects\\Symbolic Learning\\deepsym-senior-project\\runs\\train"
        self._get_exp_number()
        self.writer = SummaryWriter(os.path.join(self.results_path, f'exp{self.exp_number}'))

    def _init_dataset(self):
        if self.args.dataset == 'CRAFTER':
            # Specify the transformations
            transform = transforms.Compose([
                transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
                # Add any other transformations if needed
            ])

            # Create datasets
            train_dataset_path = r"C:\\Users\\EmirKISA\\Desktop\\Projects\\Symbolic Learning\\deepsym-senior-project\\data\\images\\train_data"
            test_dataset_path = r"C:\\Users\\EmirKISA\\Desktop\\Projects\\Symbolic Learning\\deepsym-senior-project\\data\\images\\test_data"
            val_dataset_path = r"C:\\Users\\EmirKISA\\Desktop\\Projects\\Symbolic Learning\\deepsym-senior-project\\data\\images\\val_data"

            train_dataset = AutoEncoderDataset(image_dir=train_dataset_path, transform=transform)
            test_dataset = AutoEncoderDataset(image_dir=test_dataset_path, transform=transform)
            val_dataset = AutoEncoderDataset(image_dir=val_dataset_path, transform=transform)

            self.train_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
            self.test_data = DataLoader(test_dataset, batch_size=32, shuffle=True)
            self.val_data = DataLoader(val_dataset, batch_size=32, shuffle=True)
        
    def _get_exp_number(self):
        exp_number = 1
        while True:
            if not os.path.exists(os.path.join(self.results_path, f'exp{exp_number}')):
                self.weights_dir = os.path.join(self.results_path, f'exp{exp_number}', 'weights')
                self.results_dir = os.path.join(self.results_path, f'exp{exp_number}', 'results')
                os.makedirs(self.weights_dir)
                os.makedirs(self.results_dir)
                break
            exp_number += 1

        self.exp_number = exp_number

    def loss_function(self, recon_x, x):
        MSE = F.mse_loss(recon_x, x.view(-1, 4096), reduction='mean')
        return MSE
    
    
    def weight_save(self, epoch, is_best):
        torch.save(self.model.state_dict(), os.path.join(self.results_path, f'exp{self.exp_number}', 'weights', f'last.pt'))
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.results_path, f'exp{self.exp_number}', 'weights', f'best.pt'))



    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = self.loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        avg_loss = train_loss / len(self.train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
        
        if train_loss < self.best_loss:
            self.best_loss = train_loss
            is_best = True
        else:
            is_best = False
        
        # Save the model's weights
        self.weight_save(epoch=epoch, is_best=is_best)
        # Log the loss
        self.writer.add_scalar('Training loss', loss.item(), epoch)

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


    def val(self):
        print("Validation Starts!")
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.val_loader):
                data = data.to(self.device)
                recon_batch = self.model(data)
                val_loss += self.loss_function(recon_batch, data).item()

                # Save images side by side
                for j, (original, recon) in enumerate(zip(data, recon_batch.view(-1, 3, 64, 64))):
                    # Assuming your data is normalized to [0, 1]
                    original = (original.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    recon = (recon.view(3, 64, 64).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    
                    # Concatenate and save images
                    concatenated_image = Image.fromarray(np.concatenate((original, recon), axis=1))
                    concatenated_image.save(os.path.join(self.results_dir, f"image_{i}_{j}.png"))

        val_loss /= len(self.val_loader.dataset)
        print('====> Val set loss: {:.4f}'.format(val_loss))