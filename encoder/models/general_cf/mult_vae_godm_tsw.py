import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
import numpy as np
import sys
import os

# Add the reference implementation to the path
reference_path = os.path.join(os.path.dirname(__file__), '../../../reference/PartialTSW/src')
if reference_path not in sys.path:
    sys.path.insert(0, reference_path)

from tsw.partial_tsw import PartialTSW
from tsw.utils import generate_trees_frames

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class mult_vae_GODM_TSW(BaseModel):
    def __init__(self, data_handler):
        super(mult_vae_GODM_TSW, self).__init__(data_handler)

        self.beta = self.hyper_config['beta']
        
        # TSW-specific hyperparameters
        self.ntrees = self.hyper_config.get('ntrees', 250)  # Number of trees
        self.nlines = self.hyper_config.get('nlines', 4)  # Lines per tree
        self.tsw_p = self.hyper_config.get('tsw_p', 2)  # Norm order (2 = Wasserstein-2)
        self.tsw_delta = self.hyper_config.get('tsw_delta', 2.0)  # Temperature for distance-based mass division
        self.mass_division = self.hyper_config.get('mass_division', 'distance_based')  # 'uniform' or 'distance_based'
        self.gen_mode = self.hyper_config.get('gen_mode', 'gaussian_orthogonal')  # Tree generation mode
        self.use_partial = self.hyper_config.get('use_partial', False)  # Enable partial transport (unbalanced)
        self.total_mass_src = self.hyper_config.get('total_mass_src', 1.0)  # Source distribution mass
        self.total_mass_llm = self.hyper_config.get('total_mass_llm', 1.0)  # Target distribution mass

        self.data_handler = data_handler

        # According to the original paper, the default structure is [item_num, 600, 200, 600, item_num]
        self.p_dims = [200, 600, self.item_num]
        self.q_dims = [self.item_num, 600, 200]

        # Compute the mean and variance in parallel.
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]
        )

        # Load the representations of user (or item) profiles.
        self.usrprf_embeds = torch.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = torch.tensor(configs['itmprf_embeds']).float().cuda()  # [item_num, 1536]

        self.mlp = nn.Sequential(
            nn.Linear(self.itmprf_embeds.shape[1], 600),
            nn.Tanh(),
            nn.Linear(600, 400)
        )

        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )

        self.drop = nn.Dropout(self.hyper_config['dropout'])

        # Initialize Partial TSW object
        self.tsw_obj = PartialTSW(
            ntrees=self.ntrees,
            nlines=self.nlines,
            p=self.tsw_p,
            delta=self.tsw_delta,
            mass_division=self.mass_division,
            device=configs['device']
        )

        # Generate tree frames for TSW computation (fixed during training)
        latent_dim = 200  # Latent dimension from q_dims
        self.theta, self.intercept = generate_trees_frames(
            ntrees=self.ntrees,
            nlines=self.nlines,
            d=latent_dim,
            gen_mode=self.gen_mode,
            device=configs['device']
        )

        self.final_embeds = None
        self.is_training = False

    def encode(self, x, user_emb):
        h = self.drop(x)
        # [batch, item_num] * [item_num, dim] = [batch, dim]
        hidden = torch.matmul(h, self.itmprf_embeds) + user_emb

        hidden = self.mlp(hidden)

        mu_llm = hidden[:, :200]
        logvar_llm = hidden[:, 200:]

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]

        return mu, mu_llm, logvar, logvar_llm

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def reparameterize(self, mu, logvar):
        if self.is_training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std) + mu
        else:
            z = mu
        return z

    def compute_tsw_distance(self, mu_src, logvar_src, mu_llm, logvar_llm):
        """
        Compute Tree-Sliced Wasserstein distance between two Gaussian distributions.
        
        Args:
            mu_src: Mean of source distribution (collaborative) [batch, latent_dim]
            logvar_src: Log-variance of source distribution [batch, latent_dim]
            mu_llm: Mean of target distribution (LLM) [batch, latent_dim]
            logvar_llm: Log-variance of target distribution [batch, latent_dim]
            
        Returns:
            tsw_loss: Tree-Sliced Wasserstein distance
        """
        # Sample from both distributions
        std_src = torch.exp(0.5 * logvar_src)
        std_llm = torch.exp(0.5 * logvar_llm)
        
        # Sample points from each distribution
        eps_src = torch.randn_like(mu_src)
        eps_llm = torch.randn_like(mu_llm)
        
        x_src = mu_src + eps_src * std_src  # [batch, latent_dim]
        x_llm = mu_llm + eps_llm * std_llm  # [batch, latent_dim]
        
        # Compute TSW distance
        if self.use_partial:
            # Use partial transport with unbalanced masses
            total_mass_src = torch.tensor(self.total_mass_src, device=x_src.device)
            total_mass_llm = torch.tensor(self.total_mass_llm, device=x_llm.device)
            tsw_loss = self.tsw_obj(
                x_src, 
                x_llm, 
                self.theta, 
                self.intercept,
                total_mass_X=total_mass_src,
                total_mass_Y=total_mass_llm
            )
        else:
            # Balanced transport (standard TSW)
            tsw_loss = self.tsw_obj(
                x_src, 
                x_llm, 
                self.theta, 
                self.intercept
            )
        
        return tsw_loss

    def cal_loss(self, user, batch_data):
        self.is_training = True

        user_emb = self.usrprf_embeds[user]

        mu_src, mu_llm, logvar_src, logvar_llm = self.encode(batch_data, user_emb)

        # There are two processing strategies:
        # (1) performing downstream tasks after addition, and
        # (2) directly conducting downstream tasks.
        # In practice, we found that these two strategies have their own advantages on different datasets.
        # Therefore, we adopt a unified design that performs reconstruction and matching after addition.
        mu = mu_src + mu_llm
        logvar = logvar_src + logvar_llm

        # Perform downstream tasks directly
        # mu = mu_src
        # logvar = logvar_src

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        BCE = - torch.mean(torch.sum(F.log_softmax(recon_x, 1) * batch_data, -1))

        KLD = - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # Use TSW distance instead of simple Wasserstein-2
        TSW = self.compute_tsw_distance(mu_src, logvar_src, mu_llm, logvar_llm)

        KLD = KLD + self.beta * TSW

        loss = BCE + KLD
        losses = {'rec_loss': BCE, 'reg_loss': KLD, 'tsw_loss': TSW}
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        batch_data = self.data_handler.train_data[pck_users.cpu()]
        data = torch.FloatTensor(batch_data.toarray()).to(configs['device'])
        user_emb = self.usrprf_embeds[pck_users]

        mu, mu_llm, logvar, logvar_llm = self.encode(data, user_emb)

        # If the downstream tasks are performed directly, this section should be commented out.
        mu = mu + mu_llm
        logvar = logvar + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        full_preds = self._mask_predict(recon_x, train_mask)
        return full_preds
