# -*- coding : utf-8 -*-

"""
Spatial Multi-attention Neural Process
"""
from attention_smanp import *
import torch
import torch.nn as nn


class LaplaceEncoder(nn.Module):
    """
    Lalpace Encoder [w]
    """

    def __init__(self, num_hidden):
        super(LaplaceEncoder, self).__init__()
        self.input_projection = Linear(3, num_hidden)
        self.context_projection = Linear(2, num_hidden)
        self.target_projection = Linear(num_hidden, num_hidden)

    def forward(self, context_x, context_y, target_x):
        encoder_input = torch.cat([context_x[..., 0:2], context_y],
                                  dim=-1)  # concat context location (x), context value (y)
        encoder_input = self.input_projection(encoder_input)  # (bs,nc,3)--> (bs,nc,num_hidden)
        value = encoder_input  # (bs,nc,num_hidden)
        key = self.context_projection(context_x[..., 0:2])  # (bs,nc,num_hidden)
        query = self.context_projection(target_x[..., 0:2])  # (bs,nt,num_hidden)
        query = torch.unsqueeze(query, axis=2)  # (bs,nt,2)-->(bs,nt,1,num_hidden)
        key = torch.unsqueeze(key, axis=1)  # (bs,nc,2)-->(bs,1,nc,num_hidden)
        weights = - torch.abs((key - query) * 0.5)  # [bs,nt,nc,num_hidden]
        weights = torch.sum(weights, axis=-1)  # [bs,nt,nc]
        weight = torch.softmax(weights, dim=-1)  # [bs,nt,nc]
        rep = torch.matmul(weight, value)  # (bs,nt,nc)*(bs,nc,hidden_number)=(bs,nt,hidden_number)
        rep = self.target_projection(rep)  # (bs,nt,hidden_number)

        return rep


class CrossEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """

    def __init__(self, input_dim, num_hidden):
        super(CrossEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(input_dim - 1, num_hidden)
        self.target_projection = Linear(input_dim - 1, num_hidden)

    def forward(self, context_x, context_y, target_x):
        encoder_input = torch.cat([context_x[..., 2:], context_y],
                                  dim=-1)  # concat context location (x), context value (y)
        encoder_input = self.input_projection(encoder_input)  # (bs,nc,input_dim)--> (bs,nc,num_hidden)
        query = self.target_projection(target_x[..., 2:])  # (bs,nt,input_dim-1)--> (bs,nt,num_hidden)
        keys = self.context_projection(context_x[..., 2:])  # (bs,nc,input_dim-1)--> (bs,nc,num_hidden)

        for attention in self.cross_attentions:  # cross attention layer
            query, _ = attention(keys, encoder_input, query)  # (bs,nt,hidden_number)

        return query


class LatentEncoder(nn.Module):
    """
    Latent Encoder [z]
    """

    def __init__(self, x_size, y_size, num_hidden):
        super(LatentEncoder, self).__init__()
        self.x_size = x_size + 2
        self.y_size = y_size
        self.num_hidden = num_hidden
        self.encoder = nn.Sequential(
            nn.Linear(self.x_size + self.y_size, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, self.num_hidden),
        )
        self.Linear = nn.Linear(self.num_hidden, 2)
        self.softplus = nn.Softplus()

    def forward(self, context_x, context_y, target_x):
        xy = torch.cat([context_x, context_y], dim=-1)  # (batch_size, n_context, x_size + y_size)
        bs, nc, xy_size = xy.shape  # (bs,nc, x_size)
        _, nt, x_size = target_x.shape  # (bs,nt, x_size)
        xy = xy.view((bs * nc, xy_size))  # (bs*nc, x_size + y_size)
        context_z = self.encoder(xy)  # (bs * n, num_hidden)
        context_z = context_z.view((bs, nc, self.num_hidden))  # (bs, nc, num_hidden)
        encoder = torch.mean(context_z, dim=1)  # (bs, 1,num_hidden)
        encoder = encoder.unsqueeze(dim=1).repeat((1, nt, 1))  # (bs, num_hidden)
        out = self.Linear(encoder)  # decoder cat(z,target_x, r, w)  (bs,3*num_hidden+tx)--> (bs  nt, 2)
        prior_mu = out[:, :, 0]  # prior_mean
        prior_log_sigma = out[:, :, 1]
        prior_sigma = 0.1 + 0.9 * self.softplus(prior_log_sigma)  # variance  sigma=0.1+0.9*log(1+exp(log_sigma))

        return encoder, prior_mu, prior_sigma


class Decoder(nn.Module):
    """
    Dencoder
    """

    def __init__(self, x_size, y_size, num_hidden):
        super(Decoder, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.num_hidden = num_hidden
        self.attribute = Linear(self.x_size, int(self.num_hidden / 4))
        self.location = Linear(2, int(self.num_hidden / 4))
        self.decoder = nn.Sequential(
            nn.Linear(3 * self.num_hidden + int(self.num_hidden / 2), num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 2),
        )

        self.softplus = nn.Softplus()

    def forward(self, z, target_x, r, w):
        """ context_x : (batch_size, n_context, x_size)
            context_y : (batch_size, n_context, y_size)
            target_x : (batch_size, n_target, x_size)
        """

        bs, nt, x_size = target_x.shape  # (bs,nt, x_size)
        t_x = self.attribute(target_x[..., 2:])
        s_x = self.location(target_x[..., 0:2])
        z_tx = torch.cat([t_x, s_x], dim=-1)
        z_tx = torch.cat([torch.cat([z, z_tx], dim=-1), r], dim=-1)  # (bs, nt, z_size + x_size + z_size)
        z_tx = torch.cat([z_tx, w], dim=-1)
        z_tx = z_tx.view((bs * nt, 3 * self.num_hidden + int(self.num_hidden / 2)))
        decoder = self.decoder(z_tx)  # (bs * nt, 2)
        decoder = decoder.view((bs, nt, 2))  # (bs, nt, 2)
        mu = decoder[:, :, 0]
        log_sigma = decoder[:, :, 1]
        sigma = 0.1 + 0.9 * self.softplus(log_sigma)  # variance  sigma=0.1+0.9*log(1+exp(log_sigma))
        return mu, sigma


class SpatialNeuralProcess(nn.Module):
    def __init__(self, x_size, y_size, num_hidden):
        super(SpatialNeuralProcess, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.num_hidden = num_hidden
        self.determine = CrossEncoder(self.x_size + self.y_size, self.num_hidden)
        self.location = LaplaceEncoder(num_hidden)
        self.LatenEncoder = LatentEncoder(self.x_size, self.y_size, self.num_hidden)
        self.decoder = Decoder(self.x_size, self.y_size, self.num_hidden)

    def forward(self, context_x, context_y, target_x):
        """ context_x : (batch_size, n_context, x_size)
            context_y : (batch_size, n_context, y_size)
            target_x : (batch_size, n_target, x_size)
        """
        z, prior_mu, prior_sigma = self.LatenEncoder(context_x, context_y, target_x)  # Encoder  (c_x,c_y)
        r = self.determine(context_x, context_y, target_x)  # attribute_cross_attention(c_x,c_y,t_x)
        w = self.location(context_x, context_y,
                          target_x)  # Laplace_location_attention(c_s,c_y,t_s)
        mu, sigma = self.decoder(z, target_x, r, w)  # decoder cat(t_s,t_x, z,r, w) --> (bs  nt, 2)

        return prior_mu, prior_sigma, mu, sigma


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

        self.kl_div = nn.KLDivLoss()

    def forward(self, prior_mu, prior_sigma, mu, sigma, target_y):
        """ mu : (bs, n_target)
            sigma : (bs, n_target)
            target_y : (bs, n_target)
        """
        loss = 0.0
        bs = mu.shape[0]
        nt = mu.shape[1]
        for i in range(bs):
            #
            dist1 = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu[i],
                                                                               covariance_matrix=torch.diag(
                                                                                   sigma[i]))

            dist2 = torch.distributions.multivariate_normal.MultivariateNormal(loc=prior_mu[i],
                                                                               covariance_matrix=torch.diag(
                                                                                   prior_sigma[i]))
            prior = dist2.sample().unsqueeze(0)  # Posterior sampling
            poster = dist1.sample().unsqueeze(0)  # Prior sampling
            kl_loss = self.kl_div(torch.log_softmax(poster[i], -1), torch.softmax(prior[i], -1))
            log_prob = dist1.log_prob(target_y[i])
            loss = -(log_prob/nt - kl_loss)
        loss = loss

        return loss
