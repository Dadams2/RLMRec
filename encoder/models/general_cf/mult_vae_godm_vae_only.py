import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
import numpy as np

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class mult_vae_godm_vae_only(BaseModel):
    """
    VAE-only variant of mult_vae_GODM with LLM signals removed (identity).
    This version sets mu_llm = 0 and logvar_llm = 0, effectively making the model
    rely purely on collaborative filtering signals from the VAE encoder/decoder.
    """
    def __init__(self, data_handler):
        super(mult_vae_godm_vae_only, self).__init__(data_handler)

        self.beta = self.hyper_config['beta']

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
        # Keep these for compatibility with visualization code, but won't use them in forward pass
        self.usrprf_embeds = torch.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = torch.tensor(configs['itmprf_embeds']).float().cuda()  # [item_num, 1536]

        # Keep MLP for compatibility, but won't use it
        self.mlp = nn.Sequential(
            nn.Linear(self.itmprf_embeds.shape[1], 600),
            nn.Tanh(),
            nn.Linear(600, 400)
        )

        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )

        self.drop = nn.Dropout(self.hyper_config['dropout'])

        self.final_embeds = None
        self.is_training = False

    def encode(self, x, user_emb):
        h = self.drop(x)
        
        # Compute LLM embeddings but set them to zero (identity for addition)
        # This maintains compatibility with visualization code
        batch_size = x.shape[0]
        mu_llm = torch.zeros(batch_size, 200, device=x.device)
        logvar_llm = torch.zeros(batch_size, 200, device=x.device)

        # Standard VAE encoder
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

    def cal_loss(self, user, batch_data):
        self.is_training = True

        user_emb = self.usrprf_embeds[user]

        mu_src, mu_llm, logvar_src, logvar_llm = self.encode(batch_data, user_emb)

        # Since mu_llm and logvar_llm are zeros, this is equivalent to:
        # mu = mu_src, logvar = logvar_src
        mu = mu_src + mu_llm
        logvar = logvar_src + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        BCE = - torch.mean(torch.sum(F.log_softmax(recon_x, 1) * batch_data, -1))

        KLD = - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # Remove Wasserstein Distance loss since LLM signals are identity
        # mean_diff = torch.norm(mu_src - mu_llm, dim=1) ** 2
        # var_diff = torch.norm(torch.exp(0.5 * logvar_src) - torch.exp(0.5 * logvar_llm), dim=1) ** 2
        # WD = torch.mean(torch.sqrt(mean_diff + var_diff + 10e-8))
        # KLD = KLD + self.beta * WD

        loss = BCE + KLD
        losses = {'rec_loss': BCE, 'reg_loss': KLD}
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        batch_data = self.data_handler.train_data[pck_users.cpu()]
        data = torch.FloatTensor(batch_data.toarray()).to(configs['device'])
        user_emb = self.usrprf_embeds[pck_users]

        mu, mu_llm, logvar, logvar_llm = self.encode(data, user_emb)

        # Since mu_llm and logvar_llm are zeros, this is pure VAE
        mu = mu + mu_llm
        logvar = logvar + logvar_llm

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        full_preds = self._mask_predict(recon_x, train_mask)
        return full_preds
