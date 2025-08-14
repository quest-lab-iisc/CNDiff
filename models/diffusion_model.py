import torch.nn as nn
import torch
import math

from utils.diffusion_utils import make_beta_schedule, extract
from models.model import denoiser
from models.model import condition
from utils.model_utils import StepEmbedding
    
class nonlinear_conditional_ddpm(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device = config.train.device
        self.num_timesteps = config.diff.timesteps

        #### betas and alphas for diffusion
        betas = make_beta_schedule(schedule=config.diff.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diff.beta_start, end=config.diff.beta_end)
        
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diff.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        
        #### data parameters
        self.config = config
        self.pred_len = config.data.pred_len
        self.noise_type = config.diff.noise_type
        
        #### model initialisation for condition network
        self.diffusion_model = denoiser(config)
        if config.use_cond:
            self.condition_model = condition(config)
            

        if self.noise_type == "t_phi":
            print("Using T_phi")
        
            self.t_phi = T_phi(config)
            self.time_emb = StepEmbedding(config.data.feature_dim, freq_dim=256)
    

    def mu_t_phi(self, t, batch_y, cond_info=None):

        """
        transformed output of T_phi
        """
        
        batch_y  = batch_y 
        out = self.t_phi(batch_y, self.time_emb(t).unsqueeze(1))
        
        return out


    def q_sample(self, batch_y, condition_info, t):

        """
        forward process for conditional and learnable mean 
        """
       
        sqrt_alpha_bar_t = extract(self.alphas_bar_sqrt, t, batch_y)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, batch_y)

        if self.noise_type=="t_phi":
            batch_y_trans = self.mu_t_phi(t=t, batch_y=batch_y)
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y_trans +  sqrt_one_minus_alpha_bar_t * noise

        else:
            noise = torch.randn_like(batch_y)
            y_t = sqrt_alpha_bar_t * batch_y + sqrt_one_minus_alpha_bar_t * noise

        if self.config.use_cond:
            y_t =  y_t + (1 - sqrt_alpha_bar_t) * condition_info 
        
        return y_t, noise


    def get_gammas(self,t, y_t):
        
        """
        coefficients for transformed gaussian loss
        """

        alpha_t = extract(self.alphas, t, y_t).squeeze(1).squeeze(1)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, y_t).squeeze(1).squeeze(1)
        sqrt_one_minus_alpha_bar_t_m_1 = extract(self.one_minus_alphas_bar_sqrt, t - 1, y_t).squeeze(1).squeeze(1)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()

        gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
        gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
        gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
                    sqrt_one_minus_alpha_bar_t.square())

        beta_t_hat = ((sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square())) * (1 - alpha_t)

        return gamma_0.unsqueeze(1).unsqueeze(2), gamma_1.unsqueeze(1).unsqueeze(2), gamma_2.unsqueeze(1).unsqueeze(2), sqrt_alpha_bar_t.unsqueeze(1).unsqueeze(2), beta_t_hat.unsqueeze(1).unsqueeze(2)


    def get_prior(self, batch_y, cond_info=None):

        """
        prior loss term in transformed forward process
        """

        T = torch.tensor([self.num_timesteps-1]).repeat(batch_y.shape[0]).to(self.device)
        batch_y_mean = self.mu_t_phi(t=T, batch_y=batch_y)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, T, batch_y)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
       
        u = sqrt_alpha_bar_t * batch_y_mean 

        if self.config.use_cond:
            u = u - (sqrt_alpha_bar_t) * cond_info
        
        return (1/2)*(torch.mean((u)**2, dim=(1,2)))
    
    def p_sample_loop(self, batch_y, x):

        """
        inference for diffusion model
        """

        t = torch.tensor([self.num_timesteps-1]).repeat(batch_y.shape[0]).to(self.device)
        z = torch.randn_like(batch_y)

        if self.config.use_cond:
            cond_info = self.condition_model(x)
            y_t = cond_info + z
        else:
            y_t = z
            cond_info = None
        
        for t in reversed(range(1, self.num_timesteps)):
            y_t, cond_info = self.p_sample(x,  y_t, t, cond_info)
    
        z = self.p_sample_t_1to0(x, y_t, cond_info)
        return z
    
    def p_sample(self, x, y_t, t, cond_info=None):
       
        t = torch.tensor([t]).to(self.device)
    
        alpha_t = extract(self.alphas, t, y_t)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, y_t)
        sqrt_one_minus_alpha_bar_t_m_1 = extract(self.one_minus_alphas_bar_sqrt, t - 1, y_t)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()

        # y_t_m_1 posterior mean component coefficients
        gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
        gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
        gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
            sqrt_one_minus_alpha_bar_t.square())

        if self.config.use_cond:
            y_0_reparam = self.forward(x, y_t, t, cond_info).to(self.device).detach()
        else:
            y_0_reparam = self.forward(x, y_t, t).to(self.device).detach()

        if self.noise_type == "t_phi":
            z = torch.randn_like(y_0_reparam)
            t1 = ((gamma_1*sqrt_alpha_bar_t) + gamma_0)*(self.mu_t_phi(batch_y=y_0_reparam, t=t-1) )
            t2 = (gamma_1*sqrt_alpha_bar_t)*(self.mu_t_phi(batch_y=y_0_reparam, t=t))

            y_t_m_1_hat = (gamma_1 * y_t)  - ( t2 - t1 )

        else: 
            z = torch.randn_like(y_t)
            y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y_t 

        if self.config.use_cond:
            y_t_m_1_hat = y_t_m_1_hat + gamma_2 * cond_info

        
        beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
        y_t_m_1 = y_t_m_1_hat.to(self.device) + beta_t_hat.sqrt().to(self.device) * z.to(self.device)
        
        if self.config.use_cond:
            cond_info  = self.condition_model(x)

        return y_t_m_1, cond_info


        
    def p_sample_t_1to0(self, x, y_t, cond_info):

        t = torch.tensor([0]).to(self.device)
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, y_t)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()

        if self.config.use_cond:
            y_0_reparam = self.forward(x, y_t, t, cond_info).to(self.device).detach()
            y_0_reparam = y_0_reparam 
        else:
            y_0_reparam = self.forward(x, y_t, t).to(self.device).detach()
    
        y_t_m_1 = y_0_reparam.to(self.device)
        
        return y_t_m_1

    def forward(self, x, y_t, t, cond_info=None):
        
        dec_out = self.diffusion_model(x, y_t, t, cond_info)

        return dec_out
  


class T_phi(nn.Module):

    """
    T_Phi network for Time dependent non linear transformation
    """

    def __init__(self,config):

        super().__init__()
        self.w1 = nn.Parameter(torch.empty(config.data.feature_dim,config.data.feature_dim))
        self.b1 = nn.Parameter(torch.empty(config.data.feature_dim))

        self.w2 = nn.Parameter(torch.empty(config.data.pred_len,config.data.pred_len))
        self.b2 = nn.Parameter(torch.empty(config.data.pred_len))

        self.act = nn.Tanh()

        self.init_weights(self.w1, self.b1)
        

    def init_weights(self,weight,bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)


    def forward(self,x,t_emb):
        

        out = x + t_emb
        
        out = (out.permute(0,2,1) @ self.w2.T ) + self.b2
        out = out.permute(0,2,1)

        out = ( out@self.w1.T ) + self.b1
        out = self.act(out)
        
        return out 




