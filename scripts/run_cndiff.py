import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from utils.build_model import build_model
from utils.diffusion_utils import calculate_mse_mae

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing import load_data
from train.train_non_gaussian import train

# Parse the config file path from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", nargs="?", default=None, help="Path to YAML config file")
args, unknown = parser.parse_known_args()
yaml_cfg_path = args.cfg

# Read the configuration file
pipe = ConfigPipeline(
    [
        YamlConfig(yaml_cfg_path, config_name='default', config_folder='cfg/'),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder='cfg/')
    ]
)

config = pipe.read_conf()

# seed
random.seed(config.seed)
torch.manual_seed(config.seed)
np.random.seed(config.seed)

torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.benchmark =False

# Load Data
train_data, train_dataloader = load_data(config, flag='train')
val_data, val_dataloader = load_data(config, flag='val')
test_data, test_dataloader = load_data(config, flag='test')

# Loss function
criterion = nn.MSELoss()

# Load model
model = build_model(config)
optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=config.train.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

# train model
Trainer = train(config=config, train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader,
                 optimizer=optimizer,scheduler=scheduler, criterion=criterion, model=model)

# training
if config.test:
    final_model = Trainer.model
else:
    final_model = Trainer()

# Loading the saved model
best_model_path = f'saved_models/{config.run_name}.pth'
final_model.load_state_dict(torch.load(best_model_path, weights_only = True))

# testing
full_pred, full_true = Trainer.test(final_model)

mse, mae = calculate_mse_mae(full_pred, full_true)

run_name = f"{config.run_name}_bsche{config.diff.beta_schedule}_bstar{config.diff.beta_start}_bend{config.diff.beta_end}_slen{config.data.seq_len}_plen{config.data.pred_len}_tsteps{config.diff.timesteps}_hdim{config.model.hidden_dim}_ndepth_{config.model.n_depth}_nemb_{config.model.n_emb}"

f = open(f"results.txt", 'a')
f.write(run_name + "  \t")
f.write('mse:{:.4f}, mae:{:.4f}'.format(mse, mae))
f.write('\n')
f.write('\n')
f.close()



