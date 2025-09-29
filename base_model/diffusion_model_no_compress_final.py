import scipy.io
import time
import os
from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import mat73
import yaml
import random
import time


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class SignalEncoder(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 128, kernel_sizes=[1, 3, 5, 7, 9, 11]):
        super().__init__()
        # Store the number of output channels per convolution
        self.N = out_channels
        
        # Create multiple convolutional layers with different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, 48, kernel_size=ks, padding=ks//2)
            for ks in kernel_sizes
        ])
        
        # Initialize weights using kaiming normal initialization
        for layer in self.conv_layers:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        # Apply all convolution layers and collect their outputs
        outputs = [conv(x) for conv in self.conv_layers]
        
        # Concatenate all outputs along the channel dimension
        concatenated_output = torch.cat(outputs, dim=1)  # Shape: (B, 6*N, L)
        
        # Apply the final 1x1 convolution
        #transformed_output = self.final_transform1(concatenated_output)  # Shape: (B, N, L)
        #transformed_output = F.silu(transformed_output)
        #transformed_output = self.final_transform2(transformed_output)  # Shape: (B, N, L)
        
        return concatenated_output


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)
       
class diff_CSDI(nn.Module):
    def __init__(self, config, device, length, inputdim=1):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        ).to(device)
        '''
        self.input_projection_modality_1 = Conv1d_with_init(inputdim, self.channels, 1).to(device)
        self.input_projection_modality_2 = Conv1d_with_init(inputdim, self.channels, 1).to(device)
        self.input_projection_modality_3 = Conv1d_with_init(inputdim, self.channels, 1).to(device)
        self.input_projection_modality_4 = Conv1d_with_init(inputdim, self.channels, 1).to(device)    
        '''
        
        self.input_projection_modality_1 = SignalEncoder().to(device)  
        self.input_projection_modality_2 = SignalEncoder().to(device)  
        self.input_projection_modality_3 = SignalEncoder().to(device)  
        self.input_projection_modality_4 = SignalEncoder().to(device)      
        
        
        self.output_projection1_1 = Conv1d_with_init(self.channels, self.channels, 1).to(device)
        self.output_projection1_2 = Conv1d_with_init(self.channels, 1, 1).to(device).to(device)

        self.output_projection2_1 = Conv1d_with_init(self.channels, self.channels, 1).to(device)
        self.output_projection2_2 = Conv1d_with_init(self.channels, 1, 1).to(device).to(device)
        
        self.output_projection3_1 = Conv1d_with_init(self.channels, self.channels, 1).to(device)
        self.output_projection3_2 = Conv1d_with_init(self.channels, 1, 1).to(device).to(device)
        
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    device = device,
                )
                for _ in range(config["layers"])
            ]
        )
        
        self.norm1 = torch.nn.LayerNorm( [self.channels, int(length/4)]).to(device)
        self.norm2 = torch.nn.LayerNorm( [self.channels, int(length/4)]).to(device)
        self.norm3 = torch.nn.LayerNorm( [self.channels, int(length/4)]).to(device)
        self.norm4 = torch.nn.LayerNorm( [self.channels, int(length/4)]).to(device)

    def forward(self, x, diffusion_step, mask, mode, borrow_mode):
        B, inputdim, L = x.shape
        x1 = x[:, :, 0:int(L/4)]
        x2 = x[:, :, int(L/4):int(L/2)]
        x3 = x[:, :, int(L/2):int(3*L/4)]
        x4 = x[:, :, int(3*L/4):L]
        
        x1 = self.input_projection_modality_1(x1)
        #x1 = F.relu(x1) 
        x1 = x1.reshape(B, self.channels, int(L/4))
        
        x2 = self.input_projection_modality_2(x2)
        #x2 = F.relu(x2)
        x2 = x2.reshape(B, self.channels, int(L/4))

        x3 = self.input_projection_modality_3(x3)
        #x3 = F.relu(x3)
        x3 = x3.reshape(B, self.channels, int(L/4))
        
        x4 = self.input_projection_modality_4(x4)
        #x4 = F.relu(x4)
        x4 = x4.reshape(B, self.channels, int(L/4))
        
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        x3 = self.norm3(x3)
        x4 = self.norm4(x4)
        
        x = torch.concat([x1,x2,x3,x4], dim = -1)
        
        diffusion_emb = self.diffusion_embedding(diffusion_step).squeeze(1)
        
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_emb, mask)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels,  L)
        x1 = x[:, :, 0:int(L/4)]
        x2 = x[:, :, int(L/4):int(L/2)]
        x3 = x[:, :, int(L/2):int(3*L/4)]
        x4 = x[:, :, int(3*L/4):L]

        if mode == 0:
            x1 = self.output_projection1_1(x1)  # (B,channel,L)
            x1 = F.relu(x1)
            x1 = self.output_projection1_2(x1)  # (B,1,L)
            x = x1
        if mode ==1:
            x2 = self.output_projection2_1(x2)  # (B,channel,L)
            x2 = F.relu(x2)
            x2 = self.output_projection2_2(x2)  # (B,1,L)
            x = x2
        if mode ==2:
            x3 = self.output_projection3_1(x3)  # (B,channel,L)
            x3 = F.relu(x3)
            x3 = self.output_projection3_2(x3)  # (B,1,L)
            x = x3
        if mode ==3:
            if borrow_mode == 0:
                x4 = self.output_projection1_1(x4)  # (B,channel,L)
                x4 = F.relu(x4)
                x4 = self.output_projection1_2(x4)  # (B,1,L)
                x = x4
            elif borrow_mode == 1:
                x4 = self.output_projection2_1(x4)  # (B,channel,L)
                x4 = F.relu(x4)
                x4 = self.output_projection2_2(x4)  # (B,1,L)
                x = x4
            else:
                x4 = self.output_projection3_1(x4)  # (B,channel,L)
                x4 = F.relu(x4)
                x4 = self.output_projection3_2(x4)  # (B,1,L)
                x = x4
                

        x = x.reshape(B, inputdim, int(L/4))
        return x
    
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, diffusion_embedding_dim, nheads, device):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels).to(device)
        self.cond_projection = Conv1d_with_init(channels, 2 * channels, 1).to(device)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1).to(device)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1).to(device)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels).to(device)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels).to(device)
        self.device = device


    def forward_time(self, y, base_shape, mask):
        B, channel, L = base_shape
        y = self.time_layer(y.permute(2, 0, 1), mask = mask).permute(1, 2, 0)
        y = y.reshape(B, channel, L).reshape(B, channel, L)
        return y
    
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos, d_model)
        position = torch.arange(pos).unsqueeze(1)  # Shape: [pos, 1]
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)  # Shape: [d_model // 2]

        pe[:, 0::2] = torch.sin(position * div_term)  # Broadcasting to [pos, d_model // 2]
        pe[:, 1::2] = torch.cos(position * div_term)  # Broadcasting to [pos, d_model // 2]

        return pe
    def forward(self, x, diffusion_emb, mask):
        B, channel, L = x.shape
        base_shape = x.shape
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        
        y = x + diffusion_emb
        y = self.forward_time(y, base_shape, mask.to(self.device)) # (B, Channel, L)
        y = self.mid_projection(y)  # (B,2*channel,L)
        
        time_embed = self.time_embedding(pos = L, d_model = channel).to(self.device) #(B,L,channel)
        time_embed = time_embed.unsqueeze(0).repeat(B, 1, 1).permute(0,2,1)  # Shape: [B, 2, 3]
        time_info = self.cond_projection(time_embed)  # (B,2*channel,L)
        y = y + time_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,L)
        y = self.output_projection(y) # (B,channel,2*L)
        

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
    
class CSDI_base(nn.Module):
    def __init__(self, config, device, L = 1500):
        super().__init__()
        self.device = device
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        #for param in ppg_extractor.parameters():
        #    param.requires_grad = False  # Freeze the parameters
        
        #for param in ecg_extractor.parameters():
        #    param.requires_grad = False  # Freeze the parameters

        input_dim = 1
        self.diffmodel = diff_CSDI(config_diff, device, L, input_dim)
        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        self.register_buffer('alpha_torch', alpha_torch)
        
        #******
        
        # One Condition A, B or C
        
        #******
        mask1 = torch.full((L, L), float('-inf')) # Starting with all -inf means there are no information flowing, we then define who can receive information from whom
        mask2 = torch.full((L, L), float('-inf'))
        mask3 = torch.full((L, L), float('-inf'))
        mask4 = torch.full((L, L), float('-inf'))
        # Masking in Biodiffuser controls which joint distribution the model produces, let us define three
        # groups of tokens for four modalities,A, B, C, D, the tokens are [A, B, C, D].
        Length = int(L/4) # Each modality has a length of L/4
        
        # Mask 1:
        mask1[Length:, :Length] = 0 #The B/C/D receive information from A
        mask1[:Length, :Length] = 0 #The A communicate within its own group
        mask1[Length:2*Length, Length:2*Length] = 0 #The B communicate within its own group
        mask1[2*Length:3*Length, 2*Length:3*Length] = 0 #The C communicate within its own group
        mask1[3*Length:4*Length, 3*Length:4*Length] = 0 #The D communicate within its own group

        # Mask 2:
        mask2[:Length, Length:2*Length] = 0 #A receive from B
        mask2[2*Length:4*Length, Length:2*Length] = 0 #C/D receive from B
        mask2[:Length, :Length] = 0 #The A communicate within its own group
        mask2[Length:2*Length, Length:2*Length] = 0 #The B communicate within its own group
        mask2[2*Length:3*Length, 2*Length:3*Length] = 0 #The C communicate within its own group
        mask2[3*Length:4*Length, 3*Length:4*Length] = 0 #The D communicate within its own group

        # Mask 3:
        mask3[:2*Length, 2*Length:3*Length] = 0 #A/B receive from C
        mask3[3*Length:4*Length, 2*Length:3*Length] = 0 #D receive from C
        mask3[:Length, :Length] = 0 #The A communicate within its own group
        mask3[Length:2*Length, Length:2*Length] = 0 #The B communicate within its own group
        mask3[2*Length:3*Length, 2*Length:3*Length] = 0 #The C communicate within its own group
        mask3[3*Length:4*Length, 3*Length:4*Length] = 0 #The D communicate within its own group
        
        # Mask 4: # probably will not be used since modality 4 is a place holder for self condition noise
        mask4[:3*Length, 3*Length:4*Length] = 0 #A/B/C receive from D
        mask4[:Length, :Length] = 0 #The A communicate within its own group
        mask4[Length:2*Length, Length:2*Length] = 0 #The B communicate within its own group
        mask4[2*Length:3*Length, 2*Length:3*Length] = 0 #The C communicate within its own group
        mask4[3*Length:4*Length, 3*Length:4*Length] = 0 #The D communicate within its own group
        
        #******
        
        # Two Conditions AB, AC or BC
        
        #******
        
        mask12 =  torch.full((L, L), float('-inf')) # Starting with all -inf means there are no information flowing, we then define who can receive information from whom
        mask23 = torch.full((L, L), float('-inf'))    
        mask13 = torch.full((L, L), float('-inf'))
        
        # Mask 12, mode A and B as conditions
        mask12[2*Length:4*Length, 0:2*Length] = 0 # C/D receiving A/B
        mask12[0:Length, Length:2*Length] = 0 # A receiving B
        mask12[Length:2*Length, 0:Length] = 0 # B receiving A
        mask12[:Length, :Length] = 0 #The A communicate within its own group
        mask12[Length:2*Length, Length:2*Length] = 0 #The B communicate within its own group
        mask12[2*Length:3*Length, 2*Length:3*Length] = 0 #The C communicate within its own group
        mask12[3*Length:4*Length, 3*Length:4*Length] = 0 #The D communicate within its own group
        
        # Mask 23, mode B and C as conditions
        mask23[0:Length, Length:3*Length] = 0 # A receiving B/C
        mask23[3*Length:4*Length, Length:3*Length] = 0 # D receiving B/C
        mask23[Length:2*Length, 2*Length:3*Length] = 0 # B receiving C
        mask23[2*Length:3*Length, Length:2*Length] = 0 # C receiving B
        mask23[:Length, :Length] = 0 #The A communicate within its own group
        mask23[Length:2*Length, Length:2*Length] = 0 #The B communicate within its own group
        mask23[2*Length:3*Length, 2*Length:3*Length] = 0 #The C communicate within its own group
        mask23[3*Length:4*Length, 3*Length:4*Length] = 0 #The D communicate within its own group
        
        # Mask 13, mode A and C as conditions
        mask13[Length:2*Length, 0:Length] = 0 # B receiving A
        mask13[Length:2*Length, 2*Length:3*Length] = 0 # B receiving C
        mask13[3*Length:4*Length, 0:Length] = 0 # D receiving A
        mask13[3*Length:4*Length, 2*Length:3*Length] = 0 # D receiving C
        mask13[0:Length, 2*Length:3*Length] = 0 # A receiving C
        mask13[2*Length:3*Length, 0:Length] = 0 # C receiving A 
        mask13[:Length, :Length] = 0 #The A communicate within its own group
        mask13[Length:2*Length, Length:2*Length] = 0 #The B communicate within its own group
        mask13[2*Length:3*Length, 2*Length:3*Length] = 0 #The C communicate within its own group
        mask13[3*Length:4*Length, 3*Length:4*Length] = 0 #The D communicate within its own group

        #******
        
        # Three Conditions ABC
        
        #******

        mask123 =  torch.full((L, L), float('-inf')) # Starting with all -inf means there are no information flowing, we then define who can receive information from whom
        mask123[:,0:3*Length] = 0 # D receiving A/B/C plus Talk between A/B/C, and talk within A/B/C
        mask123[3*Length:4*Length, 3*Length:4*Length] = 0 #The D communicate within its own group

        
        
        self.mask1 = mask1
        self.mask2 = mask2
        self.mask3 = mask3

        self.mask12 = mask12
        self.mask23 = mask23
        self.mask13 = mask13

        self.mask123  = mask123

    
    
    def one_condition(
        self, observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, is_train=1, set_t=-1
    ):  
        B, K, L = observed_data.shape
        device = observed_data.device
        
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(device)

        current_alpha = self.alpha_torch[t]  # (B,1,1)
        unit_length = int(L/4)
        noise = torch.randn(B, K, unit_length).to(device)
        
        cross_modality = False
        self_modality = False
        
        if task_dice > 0.5:
            cross_modality = True
        else:            
            self_modality = True
        
        if cross_modality:
            modality = random.randint(1, 3)
            train_dice = random.randint(1, 2)
            if modality == 1:
            # noisy_data_m1, A tokens as conditions
                if train_dice == 1:
                    noisy_data_B = (current_alpha ** 0.5) * observed_data[:,:,int(L/4):int(L/2)] + (1.0 - current_alpha) ** 0.5 * noise # B noise
                    noisy_data_m1 = torch.concat([observed_data[:,:,0:int(L/4)], noisy_data_B, noise, noise], dim=-1)                  
                    # Predict noise B
                    predicted = self.diffmodel(noisy_data_m1, t, self.mask1.to(self.device), mode = 1, borrow_mode = 1) # Borrow mode doesn't matter in cross modality here
                    residual = noise - predicted # (B, 1, L)
                    loss1 = (residual ** 2).sum()/L
                else:
                    noisy_data_C = (current_alpha ** 0.5) * observed_data[:,:,int(L/2):int(3*L/4)] + (1.0 - current_alpha) ** 0.5 * noise # C noise
                    noisy_data_m1 = torch.concat([observed_data[:,:,0:int(L/4)], noise, noisy_data_C, noise], dim=-1)
                    # Predict noise C
                    predicted = self.diffmodel(noisy_data_m1, t, self.mask1.to(self.device), mode = 2, borrow_mode = 2) # Borrow mode doesn't matter in cross modality here
                    residual = noise - predicted # (B, 1, L)
                    loss1 = (residual ** 2).sum()/L


            if modality == 2:
                if train_dice == 1:
                    # noisy_data_m2, B tokens as conditions
                    noisy_data_A = (current_alpha ** 0.5) * observed_data[:,:,0:int(L/4)] + (1.0 - current_alpha) ** 0.5 * noise # A noise
                    noisy_data_m2 = torch.concat([noisy_data_A, observed_data[:,:,int(L/4):int(L/2)], noise, noise], dim=-1)                  
                    #predict noise A
                    predicted = self.diffmodel(noisy_data_m2, t, self.mask2.to(self.device), mode = 0, borrow_mode = 0)
                    residual = noise - predicted # (B, 1, L)
                    loss1 = (residual ** 2).sum()/L
                else:
                    # noisy_data_m2, B tokens as conditions
                    noisy_data_C = (current_alpha ** 0.5) * observed_data[:,:,int(L/2):int(3*L/4)] + (1.0 - current_alpha) ** 0.5 * noise # C noise
                    noisy_data_m2 = torch.concat([noise, observed_data[:,:,int(L/4):int(L/2)], noisy_data_C, noise], dim=-1)                  
                    #predict noise C
                    predicted = self.diffmodel(noisy_data_m2, t, self.mask2.to(self.device), mode = 2, borrow_mode = 2)
                    residual = noise - predicted # (B, 1, L)
                    loss1 = (residual ** 2).sum()/L

    

            if modality == 3:
                if train_dice == 1:
                    # noisy_data_m3, C tokens as conditions
                    noisy_data_A = (current_alpha ** 0.5) * observed_data[:,:,0:int(L/4)] + (1.0 - current_alpha) ** 0.5 * noise # A noise
                    noisy_data_m3 = torch.concat([noisy_data_A, noise, observed_data[:,:,int(L/2):int(3*L/4)], noise], dim=-1)
                    #predict noise A
                    predicted = self.diffmodel(noisy_data_m3, t, self.mask3.to(self.device), mode = 0, borrow_mode = 0)
                    residual = noise - predicted # (B, 1, L)
                    loss1 = (residual ** 2).sum()/L
                else:
                    # noisy_data_m3, C tokens as conditions
                    noisy_data_B = (current_alpha ** 0.5) * observed_data[:,:,int(L/4):int(L/2)] + (1.0 - current_alpha) ** 0.5 * noise # B noise
                    noisy_data_m3 = torch.concat([noise, noisy_data_B, observed_data[:,:,int(L/2):int(3*L/4)], noise], dim=-1)
                    #predict noise A
                    predicted = self.diffmodel(noisy_data_m3, t, self.mask3.to(self.device), mode = 1, borrow_mode = 1)
                    residual = noise - predicted # (B, 1, L)
                    loss1 = (residual ** 2).sum()/L
                
        
        #*************************************************************************************************
        
        # Or we do self condition tasks
        
        #*************************************************************************************************        
        
        
        if dirty_dice > 0.5:
            dirty_sig = sig_impute
        else:
            dirty_sig = sig_denoise
            mask = torch.ones_like(mask)
        if self_modality:
            modality = random.randint(1, 3)
            if modality == 1:
            # noisy_data_m1, corrupted A tokens as conditions
                noisy_data_D = (current_alpha ** 0.5) * observed_data[:,:,0:int(L/4)] + (1.0 - current_alpha) ** 0.5 * noise # D noise: clean A token + noise/missingness
                noisy_data_m1 = torch.concat([dirty_sig[:,:,0:int(L/4)], noise, noise, noisy_data_D], dim=-1)
                
                # Predict noise D
                predicted = self.diffmodel(noisy_data_m1, t, self.mask1.to(self.device), mode = 3, borrow_mode = 0)
                residual = noise - predicted # (B, 1, L)
                loss1 = ((residual*mask) ** 2).sum()/L



            if modality == 2:
                # noisy_data_m2, corrupted B tokens as conditions
                noisy_data_D = (current_alpha ** 0.5) * observed_data[:,:,int(L/4):int(L/2)] + (1.0 - current_alpha) ** 0.5 * noise # C noise
                noisy_data_m2 = torch.concat([noise, dirty_sig[:,:,int(L/4):int(L/2)], noise, noisy_data_D], dim=-1)
                
                #predict noise D
                predicted = self.diffmodel(noisy_data_m2, t, self.mask2.to(self.device), mode = 3, borrow_mode = 1)
                residual = noise - predicted # (B, 1, L)
                loss1 = ((residual*mask) ** 2).sum()/L

    

            if modality == 3:
                # noisy_data_m3, C tokens as conditions
                noisy_data_D = (current_alpha ** 0.5) * observed_data[:,:,int(L/2):int(3*L/4)] + (1.0 - current_alpha) ** 0.5 * noise # B noise            
                noisy_data_m3 = torch.concat([noise, noise, dirty_sig[:,:,int(L/2):int(3*L/4)], noisy_data_D], dim=-1)
                
                #predict noise D
                predicted = self.diffmodel(noisy_data_m3, t, self.mask3.to(self.device), mode = 3, borrow_mode = 2)
                residual = noise - predicted # (B, 1, L)
                loss1 = ((residual*mask) ** 2).sum()/L
        
        return loss1
    
#************************#
    #Two conditions
#************************#
    
    def two_conditions(
            self, observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, is_train=1, set_t=-1
        ):  
            B, K, L = observed_data.shape
            device = observed_data.device
            
            if is_train != 1:  # for validation
                t = (torch.ones(B) * set_t).long().to(device)
            else:
                t = torch.randint(0, self.num_steps, [B]).to(device)

            current_alpha = self.alpha_torch[t]  # (B,1,1)
            unit_length = int(L/4)
            noise = torch.randn(B, K, unit_length).to(device)
            
            cross_modality = False
            self_modality = False
            
            if task_dice > 0.5:
                cross_modality = True
            else:            
                self_modality = True
            
            if cross_modality:
                modality = random.randint(1, 3)
                if modality == 1:
                    # noisy_data_m23, B&C tokens as conditions
                    noisy_data_A = (current_alpha ** 0.5) * observed_data[:,:,0:int(L/4)] + (1.0 - current_alpha) ** 0.5 * noise # A noise
                    noisy_data_m23 = torch.concat([noisy_data_A, observed_data[:,:,int(L/4):int(L/2)], observed_data[:,:,int(L/2):int(3*L/4)], noise], dim=-1)
                    
                    #predict noise A
                    predicted = self.diffmodel(noisy_data_m23, t, self.mask23.to(self.device), mode = 0, borrow_mode = 0)
                    residual = predicted - noise # (B, 1, L)
                    loss1 = (residual ** 2).sum()/L


                if modality == 2:               
                    # noisy_data_m13, A&C tokens as conditions
                    noisy_data_B = (current_alpha ** 0.5) * observed_data[:,:,int(L/4):int(L/2)] + (1.0 - current_alpha) ** 0.5 * noise # B noise
                    noisy_data_m13 = torch.concat([observed_data[:,:,0:int(L/4)], noisy_data_B, observed_data[:,:,int(L/2):int(3*L/4)], noise], dim=-1)
                    
                    # Predict noise B
                    predicted = self.diffmodel(noisy_data_m13, t, self.mask13.to(self.device), mode = 1, borrow_mode = 1) # Borrow mode doesn't matter in cross modality here
                    residual = predicted - noise # (B, 1, L)
                    loss1 = (residual ** 2).sum()/L
        

                if modality == 3:
                    # noisy_data_m12, A&B tokens as conditions
                    noisy_data_C = (current_alpha ** 0.5) * observed_data[:,:,int(L/2):int(3*L/4)] + (1.0 - current_alpha) ** 0.5 * noise # A noise
                    noisy_data_m12 = torch.concat([observed_data[:,:,0:int(L/4)], observed_data[:,:,int(L/4):int(L/2)], noisy_data_C, noise], dim=-1)
                    
                    #predict noise C
                    predicted = self.diffmodel(noisy_data_m12, t, self.mask12.to(self.device), mode = 2, borrow_mode = 2)
                    residual = predicted - noise # (B, 1, L)
                    loss1 = (residual ** 2).sum()/L
                    
            
            #*************************************************************************************************
            
            # Or we do self condition tasks
            
            #*************************************************************************************************        
            
            
            if dirty_dice > 0.5:
                dirty_sig = sig_impute
            else:
                dirty_sig = sig_denoise
                mask = torch.ones_like(mask)
            if self_modality:
                modality = random.randint(1, 3)
                train_dice = random.randint(1, 2)
                if modality == 1:
                    if train_dice == 1:
                        # noisy_data_m12, dirty_A&B tokens as conditions
                        noisy_data_A = (current_alpha ** 0.5) * observed_data[:,:,0:int(L/4)] + (1.0 - current_alpha) ** 0.5 * noise # A noise
                        noisy_data_m12 = torch.concat([dirty_sig[:,:,0:int(L/4)], observed_data[:,:,int(L/4):int(L/2)], noise, noisy_data_A], dim=-1)
                        
                        #predict noise A
                        predicted = self.diffmodel(noisy_data_m12, t, self.mask12.to(self.device), mode = 3, borrow_mode = 0)
                        residual = predicted - noise # (B, 1, L)
                        loss1 = ((residual*mask) ** 2).sum()/L
                    if train_dice == 2:
                        # noisy_data_m12, A&dirty_B tokens as conditions
                        noisy_data_B = (current_alpha ** 0.5) * observed_data[:,:,int(L/4):int(L/2)] + (1.0 - current_alpha) ** 0.5 * noise # A noise
                        noisy_data_m12 = torch.concat([observed_data[:,:,0:int(L/4)], dirty_sig[:,:,int(L/4):int(L/2)], noise, noisy_data_B], dim=-1)
                        
                        #predict noise B
                        predicted = self.diffmodel(noisy_data_m12, t, self.mask12.to(self.device), mode = 3, borrow_mode = 1)
                        residual = predicted - noise # (B, 1, L)
                        loss1 = ((residual*mask) ** 2).sum()/L


                if modality == 2:               
                    if train_dice == 1:
                        # noisy_data_m13, dirty_A&C tokens as conditions
                        noisy_data_A = (current_alpha ** 0.5) * observed_data[:,:,0:int(L/4)] + (1.0 - current_alpha) ** 0.5 * noise # A noise
                        noisy_data_m13 = torch.concat([dirty_sig[:,:,0:int(L/4)], noise, observed_data[:,:,int(L/2):int(3*L/4)], noisy_data_A], dim=-1)
                        
                        #predict noise A
                        predicted = self.diffmodel(noisy_data_m13, t, self.mask13.to(self.device), mode = 3, borrow_mode = 0)
                        residual = predicted - noise # (B, 1, L)
                        loss1 = ((residual*mask) ** 2).sum()/L
                    if train_dice == 2:
                        # noisy_data_m13, A&dirty_C tokens as conditions
                        noisy_data_C = (current_alpha ** 0.5) * observed_data[:,:,int(L/2):int(3*L/4)] + (1.0 - current_alpha) ** 0.5 * noise # A noise
                        noisy_data_m13 = torch.concat([observed_data[:,:,0:int(L/4)], noise, dirty_sig[:,:,int(L/2):int(3*L/4)], noisy_data_C], dim=-1)
                        
                        #predict noise C
                        predicted = self.diffmodel(noisy_data_m13, t, self.mask13.to(self.device), mode = 3, borrow_mode = 2)
                        residual = predicted - noise # (B, 1, L)
                        loss1 = ((residual*mask) ** 2).sum()/L
        

                if modality == 3:               
                    if train_dice == 1:
                        # noisy_data_m23, dirty_B&C tokens as conditions
                        noisy_data_B = (current_alpha ** 0.5) * observed_data[:,:,int(L/4):int(L/2)] + (1.0 - current_alpha) ** 0.5 * noise # A noise
                        noisy_data_m23 = torch.concat([noise, dirty_sig[:,:,int(L/4):int(L/2)], observed_data[:,:,int(L/2):int(3*L/4)], noisy_data_B], dim=-1)
                        
                        #predict noise B
                        predicted = self.diffmodel(noisy_data_m23, t, self.mask23.to(self.device), mode = 3, borrow_mode = 1)
                        residual = predicted - noise # (B, 1, L)
                        loss1 = ((residual*mask) ** 2).sum()/L
                    if train_dice == 2:
                        # noisy_data_m13, B&dirty_C tokens as conditions
                        noisy_data_C = (current_alpha ** 0.5) * observed_data[:,:,int(L/2):int(3*L/4)] + (1.0 - current_alpha) ** 0.5 * noise # A noise
                        noisy_data_m23 = torch.concat([noise, observed_data[:,:,int(L/4):int(L/2)], dirty_sig[:,:,int(L/2):int(3*L/4)], noisy_data_C], dim=-1)
                        
                        #predict noise A
                        predicted = self.diffmodel(noisy_data_m23, t, self.mask23.to(self.device), mode = 3, borrow_mode = 2)
                        residual = predicted - noise # (B, 1, L)
                        loss1 = ((residual*mask) ** 2).sum()/L
            
            return loss1
    

    def three_conditions(
            self, observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, is_train=1, set_t=-1
        ):  
            B, K, L = observed_data.shape
            device = observed_data.device
            
            if is_train != 1:  # for validation
                t = (torch.ones(B) * set_t).long().to(device)
            else:
                t = torch.randint(0, self.num_steps, [B]).to(device)

            current_alpha = self.alpha_torch[t]  # (B,1,1)
            unit_length = int(L/4)
            noise = torch.randn(B, K, unit_length).to(device)
            self_modality = True
            
            #*************************************************************************************************
            
            # Or we do self condition tasks
            
            #*************************************************************************************************        

            if dirty_dice > 0.5:
                dirty_sig = sig_impute
            else:
                dirty_sig = sig_denoise
                mask = torch.ones_like(mask)
            if self_modality:
                modality = random.randint(1, 3)                
                if modality == 1:
                        # noisy_data_m123, dirty_A&B&C tokens as conditions
                        noisy_data_A = (current_alpha ** 0.5) * observed_data[:,:,0:int(L/4)] + (1.0 - current_alpha) ** 0.5 * noise # A noise
                        noisy_data_m123 = torch.concat([dirty_sig[:,:,0:int(L/4)], observed_data[:,:,int(L/4):int(L/2)], observed_data[:,:,int(L/2):int(3*L/4)], noisy_data_A], dim=-1)
                        
                        #predict noise A
                        predicted = self.diffmodel(noisy_data_m123, t, self.mask123.to(self.device), mode = 3, borrow_mode = 0)
                        residual = predicted - noise # (B, 1, L)
                        loss1 = ((residual*mask) ** 2).sum()/L

                if modality == 2:
                        # noisy_data_m123, A&dirty_B&C tokens as conditions
                        noisy_data_B = (current_alpha ** 0.5) * observed_data[:,:,int(L/4):int(L/2)] + (1.0 - current_alpha) ** 0.5 * noise # B noise
                        noisy_data_m123 = torch.concat([observed_data[:,:,0:int(L/4)], dirty_sig[:,:,int(L/4):int(L/2)], observed_data[:,:,int(L/2):int(3*L/4)], noisy_data_B], dim=-1)
                        
                        #predict noise B
                        predicted = self.diffmodel(noisy_data_m123, t, self.mask123.to(self.device), mode = 3, borrow_mode = 1)
                        residual = predicted - noise # (B, 1, L)
                        loss1 = ((residual*mask) ** 2).sum()/L
                
                if modality == 3:
                        # noisy_data_m123, A&B&dirty_C tokens as conditions
                        noisy_data_C = (current_alpha ** 0.5) * observed_data[:,:,int(L/2):int(3*L/4)] + (1.0 - current_alpha) ** 0.5 * noise # C noise
                        noisy_data_m123 = torch.concat([observed_data[:,:,0:int(L/4)], observed_data[:,:,int(L/4):int(L/2)], dirty_sig[:,:,int(L/2):int(3*L/4)], noisy_data_C], dim=-1)
                        
                        #predict noise B
                        predicted = self.diffmodel(noisy_data_m123, t, self.mask123.to(self.device), mode = 3, borrow_mode = 2)
                        residual = predicted - noise # (B, 1, L)
                        loss1 = ((residual*mask) ** 2).sum()/L
            
            return loss1
    

    def one_condition_imputation(self, observed_data, n_samples, model_flag, borrow_mode):
        with torch.no_grad():
            B, K, L = observed_data.shape
            imputed_samples = torch.ones(B, n_samples, K, int(L/4)).to(self.device) # (B, N, 1, L)


            for i in range(n_samples):
                noise = torch.randn(B, K, int(L/4)).to(self.device)

                if model_flag[0] == '0':
                    mask = self.mask1
                    noisy_data = torch.concat([observed_data[:,:,0:int(L/4)], noise, noise, noise], dim=-1)
                elif model_flag[0] == '1':
                    mask = self.mask2
                    noisy_data = torch.concat([noise, observed_data[:,:,int(L/4):int(L/2)], noise, noise], dim=-1)
                elif model_flag[0] == '2':
                    mask = self.mask3
                    noisy_data = torch.concat([noise, noise, observed_data[:,:,int(L/2):int(3*L/4)], noise], dim=-1)

                for t in range(self.num_steps - 1, -1, -1):
                    noise_predicted = self.diffmodel(noisy_data, torch.tensor([t]).to(self.device), mask.to(self.device), mode = int(model_flag[1]), borrow_mode = borrow_mode)
                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    noise = torch.randn_like(noise_predicted)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5

                    if model_flag[0] == '0':
                        if model_flag[1] == '1':
                            noisy_data[:,:,int(L/4):int(L/2)] = coeff1 * (noisy_data[:,:,int(L/4):int(L/2)] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(L/4):int(L/2)] += sigma * noise                      
                        elif model_flag[1] == '2':
                            noisy_data[:,:,int(L/2):int(3*L/4)] = coeff1 * (noisy_data[:,:,int(L/2):int(3*L/4)] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(L/2):int(3*L/4)] += sigma * noise
                        else:
                            noisy_data[:,:,int(3*L/4):L] = coeff1 * (noisy_data[:,:,int(3*L/4):L] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(3*L/4):L] += sigma * noise


                    if model_flag[0] == '1':
                        if model_flag[1] == '0':
                            noisy_data[:,:,0:int(L/4)] = coeff1 * (noisy_data[:,:,0:int(L/4)] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,0:int(L/4)] += sigma * noise                   
                        elif model_flag[1] == '2':
                            noisy_data[:,:,int(L/2):int(3*L/4)] = coeff1 * (noisy_data[:,:,int(L/2):int(3*L/4)] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(L/2):int(3*L/4)] += sigma * noise
                        else:
                            noisy_data[:,:,int(3*L/4):L] = coeff1 * (noisy_data[:,:,int(3*L/4):L] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(3*L/4):L] += sigma * noise


                    if model_flag[0] == '2':        
                        if model_flag[1] == '0':
                            noisy_data[:,:,0:int(L/4)] = coeff1 * (noisy_data[:,:,0:int(L/4)] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,0:int(L/4)] += sigma * noise 
                        elif model_flag[1] == '1':
                            noisy_data[:,:,int(L/4):int(L/2)] = coeff1 * (noisy_data[:,:,int(L/4):int(L/2)] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(L/4):int(L/2)] += sigma * noise 
                        else:
                            noisy_data[:,:,int(3*L/4):L] = coeff1 * (noisy_data[:,:,int(3*L/4):L] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(3*L/4):L] += sigma * noise

                if model_flag[1] == '0':                
                    imputed_samples[:, i] = noisy_data[:,:,0:int(L/4)].detach()
                if model_flag[1] == '1':     
                    imputed_samples[:, i] = noisy_data[:,:,int(L/4):int(L/2)].detach()
                if model_flag[1] == '2':                
                    imputed_samples[:, i] = noisy_data[:,:,int(L/2):int(3*L/4)].detach()
                if model_flag[1] == '3':                
                    imputed_samples[:, i] = noisy_data[:,:,int(3*L/4):L].detach()
            return imputed_samples   

    
    def one_condition_DDIM_imputation(self, observed_data, n_samples, sample_steps, model_flag, borrow_mode, ratio):
        # t1 = time.time()
        with torch.no_grad():
            B, K, L = observed_data.shape
            imputed_samples = torch.ones(B, n_samples, K, int(L/4)).to(self.device) # (B, N, 1, L)
            total_steps = self.num_steps
            skip_steps = total_steps // sample_steps
            sampling_time_steps = torch.arange(0, total_steps, skip_steps).flip(0).to(self.device)
            traj = []
            for i in range(n_samples):
                noise = torch.randn(B, K, int(L/4)).to(self.device)
                if i == 0:
                    traj.append(noise)
                    

                if model_flag[0] == '0':
                    mask = self.mask1
                    noisy_data = torch.concat([observed_data[:,:,0:int(L/4)], noise, noise, noise], dim=-1)
                elif model_flag[0] == '1':
                    mask = self.mask2
                    noisy_data = torch.concat([noise, observed_data[:,:,int(L/4):int(L/2)], noise, noise], dim=-1)
                elif model_flag[0] == '2':
                    mask = self.mask3
                    noisy_data = torch.concat([noise, noise, observed_data[:,:,int(L/2):int(3*L/4)], noise], dim=-1)
                # t2 = time.time()
                # print("initialize time: ", t2-t1)
                for t_index in range(len(sampling_time_steps) - 1):
                    t = sampling_time_steps[t_index]
                    t_next = sampling_time_steps[t_index + 1] if t_index < len(sampling_time_steps) - 1 else torch.tensor(0)
                    # print("t", t)
                    # print("t_next", t_next)
                    noise_predicted = self.diffmodel(noisy_data, torch.tensor([t]).to(self.device), mask.to(self.device), mode = int(model_flag[1]), borrow_mode = borrow_mode)/ratio
                    alpha_t = self.alpha[t]
                    alpha_t_next = self.alpha[t_next]
                    # print("alpha_t", alpha_t)
                    # print("alpha_t_next", alpha_t_next)
                    # import pdb; pdb.set_trace()

                    if model_flag[0] == '0':
                        if model_flag[1] == '1':
                            x0_hat = (noisy_data[:,:,int(L/4):int(L/2)] - ((1-alpha_t)**0.5)*noise_predicted)/alpha_t**0.5
                            x_next = (alpha_t_next**0.5) * x0_hat + ((1 - alpha_t_next) **0.5)* noise_predicted
                            noisy_data[:,:,int(L/4):int(L/2)] = x_next                      
                        elif model_flag[1] == '2':
                            x0_hat = (noisy_data[:,:,int(L/2):int(3*L/4)] - ((1-alpha_t)**0.5)*noise_predicted)/alpha_t**0.5
                            x_next = (alpha_t_next**0.5) * x0_hat + ((1 - alpha_t_next) **0.5)* noise_predicted
                            noisy_data[:,:,int(L/2):int(3*L/4)] = x_next
                        else:
                            x0_hat = (noisy_data[:,:,int(3*L/4):L] - ((1-alpha_t)**0.5)*noise_predicted)/alpha_t**0.5
                            x_next = (alpha_t_next**0.5) * x0_hat + ((1 - alpha_t_next) **0.5)* noise_predicted
                            noisy_data[:,:,int(3*L/4):L] = x_next



                    if model_flag[0] == '1':
                        if model_flag[1] == '0':
                            x0_hat = (noisy_data[:,:,0:int(L/4)] - ((1-alpha_t)**0.5)*noise_predicted)/alpha_t**0.5
                            x_next = (alpha_t_next**0.5) * x0_hat + ((1 - alpha_t_next) **0.5)* noise_predicted
                            noisy_data[:,:,0:int(L/4)] = x_next
                  
                        elif model_flag[1] == '2':
                            x0_hat = (noisy_data[:,:,int(L/2):int(3*L/4)] - ((1-alpha_t)**0.5)*noise_predicted)/alpha_t**0.5
                            x_next = (alpha_t_next**0.5) * x0_hat + ((1 - alpha_t_next) **0.5)* noise_predicted
                            noisy_data[:,:,int(L/2):int(3*L/4)] = x_next
                            
                        else:
                            x0_hat = (noisy_data[:,:,int(3*L/4):L] - ((1-alpha_t)**0.5)*noise_predicted)/alpha_t**0.5
                            x_next = (alpha_t_next**0.5) * x0_hat + ((1 - alpha_t_next) **0.5)* noise_predicted
                            noisy_data[:,:,int(3*L/4):L] = x_next


                    if model_flag[0] == '2':        
                        if model_flag[1] == '0':
                            x0_hat = (noisy_data[:,:,0:int(L/4)] - ((1-alpha_t)**0.5)*noise_predicted)/alpha_t**0.5
                            x_next = (alpha_t_next**0.5) * x0_hat + ((1 - alpha_t_next) **0.5)* noise_predicted
                            noisy_data[:,:,0:int(L/4)] = x_next
                        elif model_flag[1] == '1':
                            x0_hat = (noisy_data[:,:,int(L/4):int(L/2)] - ((1-alpha_t)**0.5)*noise_predicted)/alpha_t**0.5
                            x_next = (alpha_t_next**0.5) * x0_hat + ((1 - alpha_t_next) **0.5)* noise_predicted
                            noisy_data[:,:,int(L/4):int(L/2)] = x0_hat 
                        else:
                            x0_hat = (noisy_data[:,:,int(3*L/4):L] - ((1-alpha_t)**0.5)*noise_predicted)/alpha_t**0.5
                            x_next = (alpha_t_next**0.5) * x0_hat + ((1 - alpha_t_next) **0.5)* noise_predicted
                            noisy_data[:,:,int(3*L/4):L] = x_next
                    if i ==0 :
                        traj.append(x_next)

                if model_flag[1] == '0':                
                    imputed_samples[:, i] = noisy_data[:,:,0:int(L/4)].detach()
                if model_flag[1] == '1':     
                    imputed_samples[:, i] = noisy_data[:,:,int(L/4):int(L/2)].detach()
                if model_flag[1] == '2':                
                    imputed_samples[:, i] = noisy_data[:,:,int(L/2):int(3*L/4)].detach()
                if model_flag[1] == '3':                
                    imputed_samples[:, i] = noisy_data[:,:,int(3*L/4):L].detach()
            return imputed_samples, torch.stack(traj, dim=1)
        
    
    def two_condition_imputation(self, observed_data, n_samples, model_flag, borrow_mode):
        with torch.no_grad():
            B, K, L = observed_data.shape
            imputed_samples = torch.ones(B, n_samples, K, int(L/4)).to(self.device) # (B, N, 1, L)


            for i in range(n_samples):
                noise = torch.randn(B, K, int(L/4)).to(self.device)

                if model_flag[0:2] == '01':
                    mask = self.mask12
                    noisy_data = torch.concat([observed_data[:,:,0:int(L/4)], observed_data[:,:,int(L/4):int(L/2)], noise, noise], dim=-1)
                elif model_flag[0:2] == '02':
                    mask = self.mask13
                    noisy_data = torch.concat([observed_data[:,:,0:int(L/4)], noise, observed_data[:,:,int(L/2):int(3*L/4)], noise], dim=-1)
                elif model_flag[0:2] == '12':
                    mask = self.mask23
                    noisy_data = torch.concat([noise, observed_data[:,:,int(L/4):int(L/2)], observed_data[:,:,int(L/2):int(3*L/4)], noise], dim=-1)

                for t in range(self.num_steps - 1, -1, -1):
                    noise_predicted = self.diffmodel(noisy_data, torch.tensor([t]).to(self.device), mask.to(self.device), mode = int(model_flag[2]), borrow_mode = borrow_mode)
                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    noise = torch.randn_like(noise_predicted)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5

                    if model_flag[0:2] == '01':
                        if model_flag[2] == '3':
                            noisy_data[:,:,int(3*L/4):L] = coeff1 * (noisy_data[:,:,int(3*L/4):L] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(3*L/4):L] += sigma * noise
                        else:
                            noisy_data[:,:,int(L/2):int(3*L/4)] = coeff1 * (noisy_data[:,:,int(L/2):int(3*L/4)] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(L/2):int(3*L/4)] += sigma * noise
                    
                    if model_flag[0:2] == '02':
                        if model_flag[2] == '3':
                            noisy_data[:,:,int(3*L/4):L] = coeff1 * (noisy_data[:,:,int(3*L/4):L] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(3*L/4):L] += sigma * noise
                        else:
                            noisy_data[:,:,int(L/4):int(L/2)] = coeff1 * (noisy_data[:,:,int(L/4):int(L/2)] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(L/4):int(L/2)] += sigma * noise
                                
                    if model_flag[0:2] == '12':
                        if model_flag[2] == '3':
                            noisy_data[:,:,int(3*L/4):L] = coeff1 * (noisy_data[:,:,int(3*L/4):L] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,int(3*L/4):L] += sigma * noise
                        else:
                            noisy_data[:,:,0:int(L/4)] = coeff1 * (noisy_data[:,:,0:int(L/4)] - coeff2 * noise_predicted)
                            if t > 0:
                                noisy_data[:,:,0:int(L/4)] += sigma * noise
                            

                if model_flag[2] == '0':                
                    imputed_samples[:, i] = noisy_data[:,:,0:int(L/4)].detach()
                if model_flag[2] == '1':     
                    imputed_samples[:, i] = noisy_data[:,:,int(L/4):int(L/2)].detach()
                if model_flag[2] == '2':                
                    imputed_samples[:, i] = noisy_data[:,:,int(L/2):int(3*L/4)].detach()
                if model_flag[2] == '3':                
                    imputed_samples[:, i] = noisy_data[:,:,int(3*L/4):L].detach()
            return imputed_samples   


    def noisy_data_update(self, noisy_data, noise_pred, coeff1, coeff2, model_flag):
        """Helper function to update noisy data based on model flag"""
        L = noisy_data.shape[-1]
        if model_flag[0] == '0':
            if model_flag[1] == '1':
                noisy_data[:,:,int(L/4):int(L/2)] = coeff1 * (noisy_data[:,:,int(L/4):int(L/2)] - coeff2 * noise_pred)
    
            elif model_flag[1] == '2':
                noisy_data[:,:,int(L/2):int(3*L/4)] = coeff1 * (noisy_data[:,:,int(L/2):int(3*L/4)] - coeff2 * noise_pred)
                
            else:
                noisy_data[:,:,int(3*L/4):L] = coeff1 * (noisy_data[:,:,int(3*L/4):L] - coeff2 * noise_pred)
                


        if model_flag[0] == '1':
            if model_flag[1] == '0':
                noisy_data[:,:,0:int(L/4)] = coeff1 * (noisy_data[:,:,0:int(L/4)] - coeff2 * noise_pred)
                                
            elif model_flag[1] == '2':
                noisy_data[:,:,int(L/2):int(3*L/4)] = coeff1 * (noisy_data[:,:,int(L/2):int(3*L/4)] - coeff2 * noise_pred)
                
            else:
                noisy_data[:,:,int(3*L/4):L] = coeff1 * (noisy_data[:,:,int(3*L/4):L] - coeff2 * noise_pred)
                


        if model_flag[0] == '2':        
            if model_flag[1] == '0':
                noisy_data[:,:,0:int(L/4)] = coeff1 * (noisy_data[:,:,0:int(L/4)] - coeff2 * noise_pred)
                
            elif model_flag[1] == '1':
                noisy_data[:,:,int(L/4):int(L/2)] = coeff1 * (noisy_data[:,:,int(L/4):int(L/2)] - coeff2 * noise_pred)
                
            else:
                noisy_data[:,:,int(3*L/4):L] = coeff1 * (noisy_data[:,:,int(3*L/4):L] - coeff2 * noise_pred)
        return noisy_data        


    def add_scaled_noise(self, noisy_data, noise, sigma, model_flag):
        """Helper function to add scaled noise based on model flag"""
        L = noisy_data.shape[-1]
        if model_flag[0] == '2':
            if model_flag[1] == '1':
                noisy_data[:,:,int(L/4):int(L/2)] += sigma * noise
            elif model_flag[1] == '0':
                noisy_data[:,:,0:int(L/4)] += sigma * noise
            else:
                noisy_data[:,:,int(3*L/4):L] += sigma * noise
        if model_flag[0] == '0':
            if model_flag[1] == '1':
                noisy_data[:,:,int(L/4):int(L/2)] += sigma * noise
            elif model_flag[1] == '2':
                noisy_data[:,:,int(L/2):int(3*L/4)] += sigma * noise
            else:
                noisy_data[:,:,int(3*L/4):L] += sigma * noise

        # Similar patterns for model_flag[0] == '1' and '2'
        return noisy_data
    
    def three_condition_imputation(self, observed_data, n_samples, model_flag, borrow_mode):
        with torch.no_grad():
            B, K, L = observed_data.shape
            imputed_samples = torch.ones(B, n_samples, K, int(L/4)).to(self.device) # (B, N, 1, L)


            for i in range(n_samples):
                noise = torch.randn(B, K, int(L/4)).to(self.device)

                noisy_data = torch.concat([observed_data[:,:,0:int(3*L/4)], noise], dim=-1)
                mask = self.mask123

                for t in range(self.num_steps - 1, -1, -1):
                    noise_predicted = self.diffmodel(noisy_data, torch.tensor([t]).to(self.device), mask.to(self.device), mode = int(model_flag[3]), borrow_mode = borrow_mode)
                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    noise = torch.randn_like(noise_predicted)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5

                    noisy_data[:,:,int(3*L/4):L] = coeff1 * (noisy_data[:,:,int(3*L/4):L] - coeff2 * noise_predicted)
                    if t > 0:
                        noisy_data[:,:,int(3*L/4):L] += sigma * noise
                            
                imputed_samples[:, i] = noisy_data[:,:,int(3*L/4):L].detach()  
            return imputed_samples   

    
    def one_condition_imputation_improved(self, observed_data, n_samples, model_flag, borrow_mode, rho = 7):
        """
        Performs one condition imputation using improved sampling based on EDM paper.
        Adapted for DDPM-style models using beta/alpha parameterization.
        
        Args:
            observed_data: Input data (B, K, L)  
            n_samples: Number of samples to generate
            model_flag: String indicating which modality to predict
            borrow_mode: Integer indicating which modality to borrow from
        """
        with torch.no_grad():
            B, K, L = observed_data.shape
            imputed_samples = torch.ones(B, n_samples, K, int(L/4)).to(self.device)
            unit_length = int(L/4)

            for i in range(n_samples):
                # Initialize noise
                noise = torch.randn(B, K, unit_length).to(self.device)
                
                if model_flag[0] == '0':
                    mask = self.mask1
                    noisy_data = torch.concat([observed_data[:,:,0:int(L/4)], 
                                            noise, noise, noise], dim=-1)
                elif model_flag[0] == '1':
                    mask = self.mask2
                    noisy_data = torch.concat([noise, 
                                            observed_data[:,:,int(L/4):int(L/2)], 
                                            noise, noise], dim=-1)
                elif model_flag[0] == '2':
                    mask = self.mask3
                    noisy_data = torch.concat([noise, noise, 
                                            observed_data[:,:,int(L/2):int(3*L/4)], 
                                            noise], dim=-1)

                # Improved sampling loop with Heun's method
                for t in range(self.num_steps - 1, -1, -1):
                    # Get current alphas
                    alpha_max = self.alpha_hat[-1]
                    alpha_min = self.alpha_hat[0]
                    sigma_max = torch.tensor(((1 - alpha_max) / alpha_max)**0.5, dtype=torch.float32)
                    sigma_min = torch.tensor(((1 - alpha_min) / alpha_min)**0.5, dtype=torch.float32)
                    step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=self.device)
                    t_steps = (sigma_max ** (1 / rho) + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
                    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
                    # First prediction (Euler step)
                    noise_pred = self.diffmodel(noisy_data, 
                                            torch.tensor([t]).to(self.device),
                                            mask.to(self.device), 
                                            mode=int(model_flag[1]),
                                            borrow_mode=borrow_mode)
                    
                    # Calculate coefficient for update
                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5

                    

                    # Heun's method second step (only if not last step)
                    if t > 0:
                        # Calculate intermediate point
                        x_euler = self.noisy_data_update(noisy_data, noise_pred, 
                                                coeff1, coeff2, model_flag)
                        
                        # Get second prediction
                        noise_pred2 = self.diffmodel(x_euler,
                                                torch.tensor([t-1]).to(self.device),
                                                mask.to(self.device),
                                                mode=int(model_flag[1]),
                                                borrow_mode=borrow_mode)
                        
                        # Average the two predictions
                        noise_pred = (0.95 * noise_pred + 0.05 * noise_pred2)

                    # Update data with noise
                    noisy_data = self.noisy_data_update(noisy_data, noise_pred, 
                                                coeff1, coeff2, model_flag)
                    
                    # Add scaled noise for next step if not last step
                    if t > 0:
                        noise = torch.randn_like(noise_pred)
                        sigma = ((1.0 - self.alpha[t - 1]) / 
                            (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                        noisy_data = self.add_scaled_noise(noisy_data, noise, 
                                                    sigma, model_flag)

                # Extract relevant portion based on model_flag
                if model_flag[1] == '0':
                    imputed_samples[:, i] = noisy_data[:,:,0:int(L/4)].detach()
                elif model_flag[1] == '1':
                    imputed_samples[:, i] = noisy_data[:,:,int(L/4):int(L/2)].detach()
                elif model_flag[1] == '2':
                    imputed_samples[:, i] = noisy_data[:,:,int(L/2):int(3*L/4)].detach()
                elif model_flag[1] == '3':
                    imputed_samples[:, i] = noisy_data[:,:,int(3*L/4):L].detach()

            return imputed_samples

    
        
    def trainning(self, observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, condition_dice, train_threshold, stage, is_train=1, set_t=-1):
        if stage == 1:
            loss1 = self.one_condition(observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, is_train=1, set_t=-1)
        elif stage == 2:
            if condition_dice < train_threshold:
                loss1 = self.one_condition(observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, is_train=1, set_t=-1)
            else:
                loss1 = self.two_conditions(observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, is_train=1, set_t=-1)
        elif stage ==3:
            if condition_dice < train_threshold/2:
                loss1 = self.one_condition(observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, is_train=1, set_t=-1)
            elif condition_dice < train_threshold:
                loss1 = self.two_conditions(observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, is_train=1, set_t=-1)
            else:
                loss1 = self.three_conditions(observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, is_train=1, set_t=-1)
        return loss1
    
    
    def generate(self, observed_data, n_samples, model_flag, borrow_mode, sample_steps=6, DDIM_flag=0, ratio = 1, improved = 0):
        if len(model_flag) == 2:
            if DDIM_flag == 1:
                imputed_samples = self.one_condition_DDIM_imputation(observed_data, n_samples, sample_steps, model_flag, borrow_mode, ratio)
            else:
                if improved == 1:
                    imputed_samples = self.one_condition_imputation_improved(observed_data, n_samples, model_flag, borrow_mode)
                else:
                    imputed_samples = self.one_condition_imputation(observed_data, n_samples, model_flag, borrow_mode)
        elif len(model_flag) == 3:
            imputed_samples = self.two_condition_imputation(observed_data, n_samples, model_flag, borrow_mode)
        else:
            imputed_samples = self.three_condition_imputation(observed_data, n_samples, model_flag, borrow_mode) 
        
        return imputed_samples
    
    # def forward(self, observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, condition_dice, train_threshold, stage, n_samples, model_flag, borrow_mode, is_train=1, set_t=-1, sample_steps=6, DDIM_flag=0, ratio = 1, improved = 0, train_gen_flag = 0):
    #     if train_gen_flag == 0:
    #         return self.trainning(observed_data, sig_impute, sig_denoise, mask, task_dice, dirty_dice, condition_dice, train_threshold, stage, is_train=1, set_t=-1)
    #     else:
    #         return self.generate(observed_data, n_samples, model_flag, borrow_mode, sample_steps=sample_steps, DDIM_flag=DDIM_flag, ratio = ratio, improved = improved)
    
    
    def forward(self, observed_data, **kwargs):
        """
        Unified forward method with cleaner parameter handling.
        
        Args:
            observed_data: The base data tensor (required for both modes)
            **kwargs: Mode-specific keyword arguments
            
        Returns:
            Either training loss or generated samples based on mode
        """
        # Determine operation mode
        train_gen_flag = kwargs.get('train_gen_flag', 0)
        
        if train_gen_flag == 0:
            # Training mode
            required_args = ['sig_impute', 'sig_denoise', 'mask', 'task_dice', 
                            'dirty_dice', 'condition_dice', 'train_threshold', 'stage']
            
            # Check for required arguments
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument for training mode: {arg}")
            
            return self.trainning(
                observed_data,
                kwargs['sig_impute'],
                kwargs['sig_denoise'],
                kwargs['mask'],
                kwargs['task_dice'],
                kwargs['dirty_dice'],
                kwargs['condition_dice'],
                kwargs['train_threshold'],
                kwargs['stage'],
                is_train=kwargs.get('is_train', 1),
                set_t=kwargs.get('set_t', -1)
            )
        else:
            # Generation mode
            required_args = ['n_samples', 'model_flag', 'borrow_mode']
            
            # Check for required arguments
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument for generation mode: {arg}")
            
            return self.generate(
                observed_data,
                kwargs['n_samples'],
                kwargs['model_flag'],
                kwargs['borrow_mode'],
                sample_steps=kwargs.get('sample_steps', 6),
                DDIM_flag=kwargs.get('DDIM_flag', 0),
                ratio=kwargs.get('ratio', 1),
                improved=kwargs.get('improved', 0)
            )
        
    
    
        
        
    