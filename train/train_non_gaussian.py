import numpy as np
import torch
import torch.nn as nn

import time
from tqdm import tqdm
from utils.diffusion_utils import extract

from models.model import EarlyStopping

###### preprocessing in dlinear
def instance_normalization(x, y0):
    x_mean = x[:,-1:,:] 
    x_std = torch.ones_like(x_mean)
    x_norm = (x - x_mean) / x_std
    y0_norm = (y0 - x_mean) / x_std
    return x_norm, y0_norm, x_mean, x_std

def instance_denormalization(y0, mean, std, pred_len):
    B = mean.shape[0]
    n_samples = y0.shape[0]//B
    std = torch.repeat_interleave(std, n_samples, dim=0).repeat(1, pred_len, 1)
    mean = torch.repeat_interleave(mean, n_samples, dim=0).repeat(1, pred_len, 1)
    y0 = y0 * std + mean
    return y0
    
class train(nn.Module):
    def __init__(self, config, train_dataloader, val_dataloader, test_dataloader, optimizer,scheduler, criterion, model):
        super(train, self).__init__()

        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = config.train.epochs
        
        self.model = model
        self.device = config.train.device
        self.criterion = criterion
        self.n_copies_to_test = config.diff.n_copies_to_test

        self.early_stopping = EarlyStopping(config=config,patience=self.config.train.patience, verbose=True)
        self.noise_type = config.diff.noise_type

    ############## CN_Diff loss
    def get_mu_t_phi_loss(self, pred_noise, batch_y, t, condition_info=None):
        gamma_0, gamma_1, gamma_2, sqrt_alpha_bar_t, beta_t_hat = self.model.get_gammas(t, pred_noise)

        term_1 = (gamma_1*sqrt_alpha_bar_t)*(self.model.mu_t_phi(batch_y=pred_noise, t=t) - self.model.mu_t_phi(batch_y=batch_y, t=t) )
        term_2 = ((gamma_1*sqrt_alpha_bar_t) + gamma_0)*(self.model.mu_t_phi(batch_y=batch_y, t=t-1) - self.model.mu_t_phi(batch_y=pred_noise, t=t-1) )

        diff_term = (torch.mean((term_1+term_2)**2, dim=(1,2), keepdim=True))*(1/(2*beta_t_hat))
        
        prior_term = self.model.get_prior(batch_y=batch_y, cond_info=condition_info)
        recon_term = torch.mean((pred_noise-batch_y)**2, dim=(1,2), keepdim=True)

        
        return torch.mean((diff_term +  prior_term + recon_term))  

    def train_an_epoch(self, epoch):

        self.model.train()
        train_loss = []
        pbar = tqdm(self.train_dataloader)

        # training
        for i, (batch_x, batch_y) in enumerate(pbar):

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x, batch_y, _, _ = instance_normalization(batch_x, batch_y)

            n = batch_x.size(0)
            t = torch.randint(low=1, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
            t = torch.cat([t, self.model.num_timesteps - t], dim=0)[:n]
            
            if self.config.use_cond:
                condition_info = self.model.condition_model(batch_x)
            else:
                condition_info = None
            
            # forward process
            y_t_batch, actual_noise = self.model.q_sample(batch_y, condition_info, t)
            
            pred_noise = self.model(batch_x, y_t_batch, t, condition_info)

            if self.noise_type == "t_phi":
                loss = self.get_mu_t_phi_loss(pred_noise, batch_y, t, condition_info)
            else:     
                loss = self.criterion(pred_noise, batch_y)

            train_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        average_loss_train = np.average(train_loss)

        # Validation
        self.model.eval()
        with torch.no_grad():
            val_loss = []
            pbar = tqdm(self.val_dataloader)

            for i, (batch_x, batch_y) in enumerate(pbar):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x, batch_y, _, _ = instance_normalization(batch_x, batch_y)

                n = batch_x.size(0)
                t = torch.randint(low=1, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.model.num_timesteps - t], dim=0)[:n]
                
                if self.config.use_cond:
                    condition_info = self.model.condition_model(batch_x)
                else:
                    condition_info = None
                    
                # forward process
                y_t_batch, actual_noise = self.model.q_sample(batch_y, condition_info, t)
                
                pred_noise = self.model(batch_x, y_t_batch, t, condition_info)

                if self.noise_type == "t_phi":
                    loss = self.get_mu_t_phi_loss(pred_noise, batch_y, t, condition_info)
                else:     
                    loss = self.criterion(pred_noise, batch_y)
                    
                val_loss.append(loss.item())

            average_loss_val = np.average(val_loss)
            
            self.scheduler.step(average_loss_val)

        return average_loss_train, average_loss_val
    
    def forward(self):

        start_time = time.time()        

        for epoch in range(self.epochs):

            epoch_time = time.time()
            
            train_loss, val_loss = self.train_an_epoch(epoch)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print(
                f"Epoch: {epoch + 1}| Train Loss: {train_loss:.7f}  Vali Loss: {val_loss:.7f}")
            
            
            self.early_stopping(val_loss, self.model, path=f"saved_models/")

            if self.early_stopping.early_stop:
                print("Early stopping")
                return self.early_stopping.model
                
            
        print("Total cost time: {}".format( time.time() - start_time))            

        return self.early_stopping.model
    
    def test(self, best_model):

        # Testing
        best_model.eval()
        with torch.no_grad():
            pbar = tqdm(self.test_dataloader)
            full_preds = []
            full_trues = []
            
            for i, (batch_x, batch_y) in enumerate(pbar):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
               
                batch_x, _, x_mean, x_std = instance_normalization(batch_x, batch_y)

                y_tile = batch_y.repeat(self.n_copies_to_test, 1, 1, 1)
                y_tile = y_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                x_tile = batch_x.repeat(self.n_copies_to_test, 1, 1, 1)
                x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)
               

                # forward process
                y_t_batch = self.model.p_sample_loop(y_tile, x_tile)

                y_t_batch = instance_denormalization(y_t_batch, x_mean, x_std, self.config.data.pred_len)

                full_preds.append(y_t_batch.reshape(self.config.data.test_batch_size,
                                            self.n_copies_to_test,
                                            (self.config.data.pred_len),
                                            self.config.data.feature_dim).detach().cpu().numpy())
                full_trues.append(batch_y.detach().cpu().numpy())

            preds = np.array(full_preds)
            trues = np.array(full_trues)
           
        return preds, trues
    
  



        
