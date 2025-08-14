import torch
import torch.nn as nn
import numpy as np 

from utils.model_utils import FullAttention, AttnMLP, DataEmbedding, StepEmbedding

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

####### Condition network
class condition(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.dec = nn.Linear(config.data.seq_len, config.data.pred_len)
        

    def forward(self,x):

        out = self.dec(x.permute(0,2,1)).permute(0,2,1)

        return out


######## Encoder
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_dim, d_model, n_heads, attn_dropout, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = FullAttention(d_model=d_model, n_heads=n_heads, attn_dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(d_model*mlp_ratio)
        self.mlp = AttnMLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, drop=0.1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6*d_model, bias=True)
        )
        

    def forward(self, x, c):
        """
        x: (B, num_feat, d_model), d_model=hidden_dim*2
        c: (B, hidden_dim)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_mod = modulate(self.norm1(x) , shift_msa, scale_msa) 
        x = x + gate_msa.unsqueeze(1) * self.attn(x_mod, x_mod, x_mod) 
        x_mod = modulate(self.norm2(x) , shift_mlp, scale_mlp) 
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)
        return x

######## Decoder
class Decoder(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_dim, d_model, pred_len, n_emb):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.mlp = nn.Sequential(
            DataEmbedding(d_model, d_model, n_emb-1),
            nn.Linear(d_model, pred_len)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2*d_model, bias=True)
        )

    def forward(self, x, k):
        """
        x: (B, num_feat, d_model)
        k: (B, hidden_dim)
        """
        shift, scale = self.adaLN_modulation(k).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.mlp(x)
        return x

class denoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        ####### embedding layer
        self.y_embedder = DataEmbedding(config.data.pred_len, config.model.hidden_dim, config.model.n_emb)
        self.k_embedder = StepEmbedding(config.model.hidden_dim, freq_dim=256)
        
        d_model = config.model.hidden_dim * 2
        self.blocks = nn.ModuleList([
            DiTBlock(config.model.hidden_dim, d_model, config.model.n_heads, config.model.attn_dropout, config.model.mlp_ratio)
            for _ in range(config.model.n_depth)])
        self.decoder = Decoder(config.model.hidden_dim, d_model, config.data.pred_len, config.model.n_emb)
        self.act = nn.Identity()
        self.initialize_weights()
        self.config = config
        if config.use_cond:
            self.cond_embedder = DataEmbedding(config.data.pred_len, config.model.hidden_dim, config.model.n_emb)

    def initialize_weights(self):

    
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)

    def forward(self, x, y, k, cond_info):
        """
        x: (B, context_length, num_feat)
        y: (B, prediction_length, num_feat)
        k: (B, )
        """
        
        y = self.y_embedder(y.permute(0, 2, 1))
        
        c = self.k_embedder(k)
        
        if self.config.use_cond:
            cond_info = self.cond_embedder(cond_info.permute(0, 2, 1))
            
        h = torch.cat([y, cond_info], dim=-1)
        
        for block in self.blocks:
            h = block(h, c) 
            
        out = self.decoder(h, c).permute(0, 2, 1) 
        out = self.act(out)

        return out




class EarlyStopping:
    def __init__(self, config, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.device = config.train.device
        self.run_name = config.run_name

    def __call__(self, val_loss, model, path):
        is_best = False
        score = -val_loss
        if val_loss is None:
            self.save_checkpoint(val_loss, model, path)
            print("Early Stopping due to NAN values")
            self.early_stop = True
        else:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
                self.counter = 0
                is_best = True
            return is_best

    def save_checkpoint(self, val_loss, model, path):
        self.model = model
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + f'{self.run_name}.pth')
        self.val_loss_min = val_loss