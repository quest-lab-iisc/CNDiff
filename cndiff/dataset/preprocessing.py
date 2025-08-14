import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features

class custom_dataset(Dataset):

    def __init__(self, configs, flag):

        if configs.data.seq_len == None:
            self.seq_len = 24 * 4           
            self.pred_len = 24 * 8         
        else:
            self.seq_len = configs.data.seq_len          
            self.pred_len = configs.data.pred_len

        assert flag in ['train', 'test', 'val']       
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]                

        self.train_val_test_split = configs.data.split_ratio     
        self.data_path = configs.data.data_path
        self.features = configs.data.features
        self.scale = configs.data.scale
        self.target = configs.data.target
        self.timeenc = configs.data.timeenc
        self.freq = configs.data.freq
        self.configs = configs
        self.__read_data__()

    
    def __read_data__(self):

        if self.scale:
            self.scaler = StandardScaler()

        #print(os.path.join(self.data_path))
        df_raw = pd.read_csv(os.path.join(self.data_path))

        # Data format, first row ['data',  ......(other features), target feature]

        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
    
        # Dividing into train, val, test split
        num_train = int(len(df_raw) * self.train_val_test_split[0])
        num_test  = int(len(df_raw) * self.train_val_test_split[2])
        num_vali  = int(len(df_raw) * self.train_val_test_split[1])

        if self.configs.run_name in ['etth1']:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.configs.run_name in ['ettm1']:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            border1s  = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s  = [num_train, num_train + num_vali, len(df_raw)]
        

        border1   = border1s[self.set_type]
        border2   = border2s[self.set_type]
        
        # mutlivarient to multi varient
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # normalising data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else: 
            data = df_data.values
        
        # time embedding
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # dividing data based on train, test, val
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):

        s_begin = index
        s_end   = s_begin + self.seq_len 
        r_begin = s_end
        r_end   = r_begin + self.pred_len

        seq_x   = self.data_x[s_begin:s_end]
        seq_y   = self.data_y[r_begin:r_end]
        

    
        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



def load_data(configs, flag):
    
    
    dataset = custom_dataset(
        configs=configs, flag = flag
    )
    print(flag, len(dataset))

    if flag == 'test':
        dataloader = DataLoader(
            dataset,
            batch_size=configs.data.test_batch_size,
            shuffle = False,
            drop_last = configs.data.shuffle,
            num_workers=10,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=configs.data.batch_size,
            shuffle = configs.data.shuffle,
            drop_last = configs.data.shuffle,
            num_workers=10,
        )

    return dataset, dataloader














