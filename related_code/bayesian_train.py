'''
Mar 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
Partly adapted from: https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from hyperparameters import Hyperparameters as hp
from data_load import *
from modules import *
import os
from tqdm import tqdm
from time import time
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, roc_auc_score, f1_score
from pdb import set_trace as bp


class Gaussian(object):
  def __init__(self, mu, rho):
    super().__init__()
    self.mu = mu
    self.rho = rho
    self.normal = torch.distributions.Normal(0,1)
  
  @property
  def sigma(self):
    return torch.log1p(torch.exp(self.rho))
  
  def sample(self):
    epsilon = self.normal.sample(self.rho.size()).to(hp.device)
    return self.mu + self.sigma * epsilon
  
  def log_prob(self, input):
    return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma) - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussian(object):
  def __init__(self, pi, sigma1, sigma2):
    super().__init__()
    self.pi = pi
    self.sigma1 = torch.Tensor([sigma1]).to(hp.device)
    self.sigma2 = torch.Tensor([sigma2]).to(hp.device)
    self.gaussian1 = torch.distributions.Normal(0, self.sigma1)
    self.gaussian2 = torch.distributions.Normal(0, self.sigma2)
    
  def log_prob(self, input):
    prob1 = torch.exp(self.gaussian1.log_prob(input))
    prob2 = torch.exp(self.gaussian2.log_prob(input))
    return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    # Weight parameters
    self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
    self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
    self.weight = Gaussian(self.weight_mu, self.weight_rho)
    # Bias parameters
    self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
    self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
    self.bias = Gaussian(self.bias_mu, self.bias_rho)
    # Distributions
    self.weight_prior = ScaleMixtureGaussian(hp.pi, hp.sigma1, hp.sigma2)
    self.bias_prior = ScaleMixtureGaussian(hp.pi, hp.sigma1, hp.sigma2)
    self.log_prior = 0
    self.log_variational_posterior = 0

  def forward(self, input, sample=False, calculate_log_probs=False):
    if self.training or sample:
      weight = self.weight.sample()
      bias = self.bias.sample()
    else:
      weight = self.weight.mu
      bias = self.bias.mu
    if self.training or calculate_log_probs:
      self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
      self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
    else:
      self.log_prior, self.log_variational_posterior = 0, 0

    return F.linear(input, weight, bias)


class BayesianEmbedding(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, padding_idx=0):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.padding_idx = padding_idx
    # Weight parameters
    self.weight_mu = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim).uniform_(-0.2, 0.2))
    self.weight_rho = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim).uniform_(-5,-4))
    with torch.no_grad():
      self.weight_mu[self.padding_idx].fill_(0)
      self.weight_rho[self.padding_idx].fill_(0)
    self.weight = Gaussian(self.weight_mu, self.weight_rho)
    # Distributions
    self.weight_prior = ScaleMixtureGaussian(hp.pi, hp.sigma1, hp.sigma2)
    self.log_prior = 0
    self.log_variational_posterior = 0

  def forward(self, input, sample=False, calculate_log_probs=False):
    if self.training or sample:
      weight = self.weight.sample()
    else:
      weight = self.weight.mu
    if self.training or calculate_log_probs:
      self.log_prior = self.weight_prior.log_prob(weight)
      self.log_variational_posterior = self.weight.log_prob(weight)
    else:
      self.log_prior, self.log_variational_posterior = 0, 0

    return F.embedding(input, weight, self.padding_idx)


class BayesianAttention(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.linear_hidden = BayesianLinear(embedding_dim, embedding_dim)
    # Context parameters
    self.context_mu = nn.Parameter(torch.Tensor(embedding_dim).uniform_(-0.2, 0.2))
    self.context_rho = nn.Parameter(torch.Tensor(embedding_dim).uniform_(-5,-4))
    self.context = Gaussian(self.context_mu, self.context_rho)
    # Distributions
    self.context_prior = ScaleMixtureGaussian(hp.pi, hp.sigma1, hp.sigma2)
    self.log_prior = 0
    self.log_variational_posterior = 0

  def forward(self, inputs, mask, sample=False, calculate_log_probs=False):
    # Hidden representation of embeddings (no change in dimensions)
    hidden = torch.tanh(self.linear_hidden(inputs, sample))  
    # Compute weight of each embedding
    if self.training or sample:
      context = self.context.sample()
    else:
      context = self.context.mu
    importance = torch.sum(hidden * context, dim=-1)
    importance = importance.masked_fill(mask == 0, -1e9)    
    # Softmax so that weights sum up to one
    attention_weights = F.softmax(importance, dim=-1)
    # Weighted sum of embeddings
    weighted_projection = inputs * torch.unsqueeze(attention_weights, dim=-1)
    # Output
    outputs = torch.sum(weighted_projection, dim=-2)
    
    if self.training or calculate_log_probs:
      self.log_prior = self.context_prior.log_prob(context) + self.linear_hidden.log_prior
      self.log_variational_posterior = self.context.log_prob(context) + self.linear_hidden.log_variational_posterior
    else:
      self.log_prior, self.log_variational_posterior = 0, 0

    return outputs, attention_weights 


class BayesianNetwork(nn.Module):
  def __init__(self, num_static, num_dp_codes, num_cp_codes, pos_weight, num_batches, output_activations=False):
    super().__init__()
    self.pos_weight = pos_weight
    self.num_batches = num_batches
    self.output_activations = output_activations

    # Embedding dimensions
    self.embed_dp_dim = int(2*np.ceil(num_dp_codes**0.25))
    self.embed_cp_dim = int(2*np.ceil(num_cp_codes**0.25))

    # Embedding layers
    self.embed_dp = BayesianEmbedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
    self.embed_cp = BayesianEmbedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)
    
    # Attention layers
    self.attention_dp = BayesianAttention(embedding_dim=self.embed_dp_dim+1) #+1 for the concatenated time
    self.attention_cp = BayesianAttention(embedding_dim=self.embed_cp_dim+1)
    
    # Fully connected output
    self.fc_dp  = BayesianLinear(self.embed_dp_dim+1, 1)
    self.fc_cp  = BayesianLinear(self.embed_cp_dim+1, 1)
    self.fc_all = BayesianLinear(num_static + 2, 1)
    
  def forward(self, stat, dp, cp, dp_t, cp_t, sample=False):
    # Embedding
    ## output dim: batch_size x seq_len x embedding_dim
    embedded_dp = self.embed_dp(dp, sample)
    embedded_cp = self.embed_cp(cp, sample)
    
    # Concatate with time
    ## output dim: batch_size x seq_len x (embedding_dim+1)
    concat_dp = torch.cat((embedded_dp, torch.unsqueeze(dp_t, dim=-1)), dim=-1)
    concat_cp = torch.cat((embedded_cp, torch.unsqueeze(cp_t, dim=-1)), dim=-1)
    
    # Attention
    ## output dim: batch_size x (embedding_dim+1)
    attended_dp, weights_dp = self.attention_dp(concat_dp, (dp > 0).float(), sample)
    attended_cp, weights_cp = self.attention_cp(concat_cp, (cp > 0).float(), sample)
    
    # Scores
    score_dp = self.fc_dp(attended_dp, sample)
    score_cp = self.fc_cp(attended_cp, sample)

    # Concatenate to variable collection
    all = torch.cat((stat, score_dp, score_cp), dim=1)
    
    # Final linear projection
    out = self.fc_all(all, sample).squeeze()

    return out, [weights_dp, weights_cp, all]

  def log_prior(self):
    return (self.embed_dp.log_prior + self.embed_cp.log_prior +
            self.attention_dp.log_prior + self.attention_cp.log_prior +
            self.fc_dp.log_prior + self.fc_cp.log_prior + self.fc_all.log_prior)
  
  def log_variational_posterior(self):
    return (self.embed_dp.log_variational_posterior + self.embed_cp.log_variational_posterior +
            self.attention_dp.log_variational_posterior + self.attention_cp.log_variational_posterior +
            self.fc_dp.log_variational_posterior + self.fc_cp.log_variational_posterior + self.fc_all.log_variational_posterior)    
  
  def sample_elbo(self, stat, dp, cp, dp_t, cp_t, label, samples=hp.samples):
    outputs = torch.zeros(samples, hp.batch_size).to(hp.device)
    log_priors = torch.zeros(samples).to(hp.device)
    log_variational_posteriors = torch.zeros(samples).to(hp.device)
    for i in range(samples):
      outputs[i], _ = self(stat, dp, cp, dp_t, cp_t, sample=True)
      log_priors[i] = self.log_prior()
      log_variational_posteriors[i] = self.log_variational_posterior()
    log_prior = log_priors.mean()
    log_variational_posterior = log_variational_posteriors.mean()
    negative_log_likelihood = F.binary_cross_entropy_with_logits(outputs.mean(0), label, reduction='sum', pos_weight=self.pos_weight)
    loss = (log_variational_posterior - log_prior)/self.num_batches + negative_log_likelihood
    return loss, log_prior, log_variational_posterior, negative_log_likelihood


if __name__ == '__main__':
  # Load data
  print('Load data...')
  data = np.load(hp.data_dir + 'data_arrays.npz')
  
  # Training and validation data
  if hp.all_train:
    trainloader, num_batches, pos_weight = get_trainloader(data, 'ALL')
  else:
    trainloader, num_batches, pos_weight = get_trainloader(data, 'TRAIN')
  
  # Vocabulary sizes
  num_static = num_static(data)
  num_dp_codes, num_cp_codes = vocab_sizes(data)
    
  print('-----------------------------------------')
  print('Train...')

  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device('cuda:0' if use_cuda else 'cpu')
  hp.device = device
  torch.backends.cudnn.benchmark = True

  # Network
  net = BayesianNetwork(num_static, num_dp_codes, num_cp_codes, pos_weight, num_batches).to(device)

  # Loss function and optimizer
  optimizer = optim.Adam(net.parameters(), lr = 0.001)  

  # Create log dir
  logdir = hp.logdir + 'bayesian_all/' if hp.all_train else hp.logdir + 'bayesian/'
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  
  # Best score on validation data
  best = 1e10
  num_bad_epochs = 0
  
  # Train
  for epoch in range(10000): 
    running_loss = 0
    print('-----------------------------------------')
    print('Epoch: {}; Bad epochs: {}'.format(epoch, num_bad_epochs))
    net.train()
    for i, (stat, dp, cp, dp_t, cp_t, label) in enumerate(tqdm(trainloader), 0):
      # move to GPU if available
      stat  = stat.to(device)
      dp    = dp.to(device)
      cp    = cp.to(device)
      dp_t  = dp_t.to(device)
      cp_t  = cp_t.to(device)
      label = label.to(device)
    
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(stat, dp, cp, dp_t, cp_t, label)
      loss.backward()
      optimizer.step()
      
      running_loss += loss.item()
    print(running_loss)

    # early stopping
    if running_loss < best:
      print('############### Saving good model ###############################')
      torch.save(net.state_dict(), logdir + 'final_model.pt')
      best = running_loss
      num_bad_epochs = 0
    else:
      num_bad_epochs = num_bad_epochs + 1
      if num_bad_epochs == hp.patience:
        break
  print('Done')
  
