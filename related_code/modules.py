'''
Nov 2018 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from hyperparameters import Hyperparameters as hp
from pdb import set_trace as bp
from modules_ode import *


class Attention(torch.nn.Module):
  """
  Dot-product attention module.
  
  Args:
    inputs: A `Tensor` with embeddings in the last dimension.
    mask: A `Tensor`. Dimensions are the same as inputs but without the embedding dimension.
      Values are 0 for 0-padding in the input and 1 elsewhere.

  Returns:
    outputs: The input `Tensor` whose embeddings in the last dimension have undergone a weighted average.
      The second-last dimension of the `Tensor` is removed.
    attention_weights: weights given to each embedding.
  """
  def __init__(self, embedding_dim):
    super(Attention, self).__init__()
    self.context = nn.Parameter(torch.Tensor(embedding_dim)) # context vector
    self.linear_hidden = nn.Linear(embedding_dim, embedding_dim)
    self.reset_parameters()
    
  def reset_parameters(self):
    nn.init.normal_(self.context)

  def forward(self, inputs, mask):
    # Hidden representation of embeddings (no change in dimensions)
    hidden = torch.tanh(self.linear_hidden(inputs))
    # Compute weight of each embedding
    importance = torch.sum(hidden * self.context, dim=-1)
    importance = importance.masked_fill(mask == 0, -1e9)
    # Softmax so that weights sum up to one
    attention_weights = F.softmax(importance, dim=-1)
    # Weighted sum of embeddings
    weighted_projection = inputs * torch.unsqueeze(attention_weights, dim=-1)
    # Output
    outputs = torch.sum(weighted_projection, dim=-2)
    return outputs, attention_weights


class GRUExponentialDecay(nn.Module):
  """
  GRU RNN module where the hidden state decays exponentially
  (see e.g. Che et al. 2018, Recurrent Neural Networks for Multivariate Time Series
  with Missing Values).
  
  Args:
    inputs: A `Tensor` with embeddings in the last dimension.
    times: A `Tensor` with the same shape as inputs containing the recorded times (but no embedding dimension).

  Returns:
    outs: Hidden states of the RNN.
  """
  def __init__(self, input_size, hidden_size, bias=True):
    super(GRUExponentialDecay, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.gru_cell = nn.GRUCell(input_size, hidden_size)
    self.decays = nn.Parameter(torch.Tensor(hidden_size)) # exponential decays vector
  
  def forward(self, inputs, times):
    # initializing and then calling cuda() later isn't working for some reason
    if torch.cuda.is_available():
      hn = torch.zeros(inputs.size(0), self.hidden_size).cuda() # batch_size x hidden_size
      outs = torch.zeros(inputs.size(0), inputs.size(1), self.hidden_size).cuda() # batch_size x seq_len x hidden_size
    else:
      hn = torch.zeros(inputs.size(0), self.hidden_size) # batch_size x hidden_size
      outs = torch.zeros(inputs.size(0), inputs.size(1), self.hidden_size) # batch_size x seq_len x hidden_size
    
    # this is slow
    for seq in range(inputs.size(1)):
      hn = self.gru_cell(inputs[:,seq,:], hn)
      outs[:,seq,:] = hn
      hn = hn*torch.exp(-torch.clamp(torch.unsqueeze(times[:,seq], dim=-1)*self.decays, min=0))
    return outs


class GRUOdeDecay(nn.Module):
  """
  GRU RNN module where the hidden state decays according to an ODE.
  (see Rubanova et al. 2019, Latent ODEs for Irregularly-Sampled Time Series)
  
  Args:
    inputs: A `Tensor` with embeddings in the last dimension.
    times: A `Tensor` with the same shape as inputs containing the recorded times (but no embedding dimension).

  Returns:
    outs: Hidden states of the RNN.
  """
  def __init__(self, input_size, hidden_size, bias=True):
    super(GRUOdeDecay, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.gru_cell = nn.GRUCell(input_size, hidden_size)
    self.decays = nn.Parameter(torch.Tensor(hidden_size)) # exponential decays vector
    
    # ODE
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.ode_net = ODENet(self.device, self.input_size, self.input_size, output_dim=self.input_size, augment_dim=0, time_dependent=False, non_linearity='softplus', tol=1e-3, adjoint=True)
  
  def forward(self, inputs, times):
    # initializing and then calling cuda() later isn't working for some reason
    if torch.cuda.is_available():
      hn = torch.zeros(inputs.size(0), self.hidden_size).cuda() # batch_size x hidden_size
      outs = torch.zeros(inputs.size(0), inputs.size(1), self.hidden_size).cuda() # batch_size x seq_len x hidden_size
    else:
      hn = torch.zeros(inputs.size(0), self.hidden_size) # batch_size x hidden_size
      outs = torch.zeros(inputs.size(0), inputs.size(1), self.hidden_size) # batch_size x seq_len x hidden_size

    # this is slow
    for seq in range(inputs.size(1)):
      hn = self.gru_cell(inputs[:,seq,:], hn)
      outs[:,seq,:] = hn
      
      times_unique, inverse_indices = torch.unique(times[:,seq], sorted=True, return_inverse=True)
      if times_unique.size(0) > 1:
        hn = self.ode_net(hn, times_unique)
        hn = hn[inverse_indices, torch.arange(0, inverse_indices.size(0)), :]
    return outs


def abs_time_to_delta(times):
  delta = torch.cat((torch.unsqueeze(times[:, 0], dim=-1), times[:, 1:] - times[:, :-1]), dim=1)
  delta = torch.clamp(delta, min=0)
  return delta


if hp.net_variant == 'birnn_concat_time_delta':
  # GRU
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(np.ceil(num_dp_codes**0.25))
      self.embed_cp_dim = int(np.ceil(num_cp_codes**0.25))

      # Embedding layers
      self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
      self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)

      # GRU layers
      self.gru_dp_fw = nn.GRU(input_size=self.embed_dp_dim+1, hidden_size=self.embed_dp_dim+1, num_layers=1, batch_first=True)
      self.gru_cp_fw = nn.GRU(input_size=self.embed_cp_dim+1, hidden_size=self.embed_cp_dim+1, num_layers=1, batch_first=True)
      self.gru_dp_bw = nn.GRU(input_size=self.embed_dp_dim+1, hidden_size=self.embed_dp_dim+1, num_layers=1, batch_first=True)
      self.gru_cp_bw = nn.GRU(input_size=self.embed_cp_dim+1, hidden_size=self.embed_cp_dim+1, num_layers=1, batch_first=True)
      
      # Fully connected output
      self.fc_dp  = nn.Linear(2*(self.embed_dp_dim+1), 1)
      self.fc_cp  = nn.Linear(2*(self.embed_cp_dim+1), 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Compute time delta
      ## output dim: batch_size x seq_len
      dp_t_delta_fw = abs_time_to_delta(dp_t)
      cp_t_delta_fw = abs_time_to_delta(cp_t)
      dp_t_delta_bw = abs_time_to_delta(torch.flip(dp_t, [1]))
      cp_t_delta_bw = abs_time_to_delta(torch.flip(cp_t, [1]))    
    
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp_fw = self.embed_dp(dp)
      embedded_cp_fw = self.embed_cp(cp)
      embedded_dp_bw = torch.flip(embedded_dp_fw, [1])
      embedded_cp_bw = torch.flip(embedded_cp_fw, [1])
      
      # Concatate with time
      ## output dim: batch_size x seq_len x (embedding_dim+1)
      concat_dp_fw = torch.cat((embedded_dp_fw, torch.unsqueeze(dp_t_delta_fw, dim=-1)), dim=-1)
      concat_cp_fw = torch.cat((embedded_cp_fw, torch.unsqueeze(cp_t_delta_fw, dim=-1)), dim=-1)
      concat_dp_bw = torch.cat((embedded_dp_bw, torch.unsqueeze(dp_t_delta_bw, dim=-1)), dim=-1)
      concat_cp_bw = torch.cat((embedded_cp_bw, torch.unsqueeze(cp_t_delta_bw, dim=-1)), dim=-1)
      ## Dropout
      concat_dp_fw = self.dropout(concat_dp_fw)
      concat_cp_fw = self.dropout(concat_cp_fw)
      concat_dp_bw = self.dropout(concat_dp_bw)
      concat_cp_bw = self.dropout(concat_cp_bw)
      
      # GRU
      ## output dim rnn:        batch_size x seq_len x (embedding_dim+1)
      ## output dim rnn_hidden: batch_size x 1 x (embedding_dim+1)
      rnn_dp_fw, rnn_hidden_dp_fw = self.gru_dp_fw(concat_dp_fw)
      rnn_cp_fw, rnn_hidden_cp_fw = self.gru_cp_fw(concat_cp_fw)
      rnn_dp_bw, rnn_hidden_dp_bw = self.gru_dp_bw(concat_dp_bw)
      rnn_cp_bw, rnn_hidden_cp_bw = self.gru_cp_bw(concat_cp_bw)      
      ## output dim rnn_hidden: batch_size x (embedding_dim+1)
      rnn_hidden_dp_fw = rnn_hidden_dp_fw.view(-1, self.embed_dp_dim+1)
      rnn_hidden_cp_fw = rnn_hidden_cp_fw.view(-1, self.embed_cp_dim+1)
      rnn_hidden_dp_bw = rnn_hidden_dp_bw.view(-1, self.embed_dp_dim+1)
      rnn_hidden_cp_bw = rnn_hidden_cp_bw.view(-1, self.embed_cp_dim+1)
      ## concatenate forward and backward: batch_size x 2*(embedding_dim+1)
      rnn_hidden_dp = torch.cat((rnn_hidden_dp_fw, rnn_hidden_dp_bw), dim=-1)
      rnn_hidden_cp = torch.cat((rnn_hidden_cp_fw, rnn_hidden_cp_bw), dim=-1)
      
      # Scores
      score_dp = self.fc_dp(self.dropout(rnn_hidden_dp))
      score_cp = self.fc_cp(self.dropout(rnn_hidden_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []


elif hp.net_variant == 'birnn_concat_time_delta_attention':
  # GRU
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(np.ceil(num_dp_codes**0.25))
      self.embed_cp_dim = int(np.ceil(num_cp_codes**0.25))

      # Embedding layers
      self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
      self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)

      # GRU layers
      self.gru_dp_fw = nn.GRU(input_size=self.embed_dp_dim+1, hidden_size=self.embed_dp_dim+1, num_layers=1, batch_first=True)
      self.gru_cp_fw = nn.GRU(input_size=self.embed_cp_dim+1, hidden_size=self.embed_cp_dim+1, num_layers=1, batch_first=True)
      self.gru_dp_bw = nn.GRU(input_size=self.embed_dp_dim+1, hidden_size=self.embed_dp_dim+1, num_layers=1, batch_first=True)
      self.gru_cp_bw = nn.GRU(input_size=self.embed_cp_dim+1, hidden_size=self.embed_cp_dim+1, num_layers=1, batch_first=True)

      # Attention layers
      self.attention_dp = Attention(embedding_dim=2*(self.embed_dp_dim+1)) #+1 for the concatenated time
      self.attention_cp = Attention(embedding_dim=2*(self.embed_cp_dim+1))
            
      # Fully connected output
      self.fc_dp  = nn.Linear(2*(self.embed_dp_dim+1), 1)
      self.fc_cp  = nn.Linear(2*(self.embed_cp_dim+1), 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Compute time delta
      ## output dim: batch_size x seq_len
      dp_t_delta_fw = abs_time_to_delta(dp_t)
      cp_t_delta_fw = abs_time_to_delta(cp_t)
      dp_t_delta_bw = abs_time_to_delta(torch.flip(dp_t, [1]))
      cp_t_delta_bw = abs_time_to_delta(torch.flip(cp_t, [1]))    
    
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp_fw = self.embed_dp(dp)
      embedded_cp_fw = self.embed_cp(cp)
      embedded_dp_bw = torch.flip(embedded_dp_fw, [1])
      embedded_cp_bw = torch.flip(embedded_cp_fw, [1])
      
      # Concatate with time
      ## output dim: batch_size x seq_len x (embedding_dim+1)
      concat_dp_fw = torch.cat((embedded_dp_fw, torch.unsqueeze(dp_t_delta_fw, dim=-1)), dim=-1)
      concat_cp_fw = torch.cat((embedded_cp_fw, torch.unsqueeze(cp_t_delta_fw, dim=-1)), dim=-1)
      concat_dp_bw = torch.cat((embedded_dp_bw, torch.unsqueeze(dp_t_delta_bw, dim=-1)), dim=-1)
      concat_cp_bw = torch.cat((embedded_cp_bw, torch.unsqueeze(cp_t_delta_bw, dim=-1)), dim=-1)
      ## Dropout
      concat_dp_fw = self.dropout(concat_dp_fw)
      concat_cp_fw = self.dropout(concat_cp_fw)
      concat_dp_bw = self.dropout(concat_dp_bw)
      concat_cp_bw = self.dropout(concat_cp_bw)
      
      # GRU
      ## output dim rnn:        batch_size x seq_len x (embedding_dim+1)
      ## output dim rnn_hidden: batch_size x 1 x (embedding_dim+1)
      rnn_dp_fw, rnn_hidden_dp_fw = self.gru_dp_fw(concat_dp_fw)
      rnn_cp_fw, rnn_hidden_cp_fw = self.gru_cp_fw(concat_cp_fw)
      rnn_dp_bw, rnn_hidden_dp_bw = self.gru_dp_bw(concat_dp_bw)
      rnn_cp_bw, rnn_hidden_cp_bw = self.gru_cp_bw(concat_cp_bw)      
      # concatenate forward and backward
      ## output dim: batch_size x seq_len x 2*(embedding_dim+1)
      rnn_dp = torch.cat((rnn_dp_fw, torch.flip(rnn_dp_bw, [1])), dim=-1)
      rnn_cp = torch.cat((rnn_cp_fw, torch.flip(rnn_cp_bw, [1])), dim=-1)

      # Attention
      ## output dim: batch_size x 2*(embedding_dim+1)
      attended_dp, weights_dp = self.attention_dp(rnn_dp, (dp > 0).float())
      attended_cp, weights_cp = self.attention_cp(rnn_cp, (cp > 0).float())
      
      # Scores
      score_dp = self.fc_dp(self.dropout(attended_dp))
      score_cp = self.fc_cp(self.dropout(attended_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []


elif hp.net_variant == 'birnn_time_decay':
  # GRU
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(np.ceil(num_dp_codes**0.25))+1
      self.embed_cp_dim = int(np.ceil(num_cp_codes**0.25))+1

      # Embedding layers
      self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
      self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)

      # GRU layers
      self.gru_dp_fw = GRUExponentialDecay(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim)
      self.gru_cp_fw = GRUExponentialDecay(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim)
      self.gru_dp_bw = GRUExponentialDecay(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim)
      self.gru_cp_bw = GRUExponentialDecay(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim)
      
      # Fully connected output
      self.fc_dp  = nn.Linear(2*self.embed_dp_dim, 1)
      self.fc_cp  = nn.Linear(2*self.embed_cp_dim, 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Compute time delta
      ## output dim: batch_size x seq_len
      dp_t_delta_fw = abs_time_to_delta(dp_t)
      cp_t_delta_fw = abs_time_to_delta(cp_t)
      dp_t_delta_bw = abs_time_to_delta(torch.flip(dp_t, [1]))
      cp_t_delta_bw = abs_time_to_delta(torch.flip(cp_t, [1]))    
    
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp_fw = self.embed_dp(dp)
      embedded_cp_fw = self.embed_cp(cp)
      embedded_dp_bw = torch.flip(embedded_dp_fw, [1])
      embedded_cp_bw = torch.flip(embedded_cp_fw, [1])
      ## Dropout
      embedded_dp_fw = self.dropout(embedded_dp_fw)
      embedded_cp_fw = self.dropout(embedded_cp_fw)
      embedded_dp_bw = self.dropout(embedded_dp_bw)
      embedded_cp_bw = self.dropout(embedded_cp_bw)
      
      # GRU
      ## output dim rnn:        batch_size x seq_len x embedding_dim
      rnn_dp_fw = self.gru_dp_fw(embedded_dp_fw, dp_t_delta_fw)
      rnn_cp_fw = self.gru_cp_fw(embedded_cp_fw, cp_t_delta_fw)
      rnn_dp_bw = self.gru_dp_bw(embedded_dp_bw, dp_t_delta_bw)
      rnn_cp_bw = self.gru_cp_bw(embedded_cp_bw, cp_t_delta_bw)      
      ## output dim rnn_hidden: batch_size x embedding_dim
      rnn_dp_fw = rnn_dp_fw[:,-1,:]
      rnn_cp_fw = rnn_cp_fw[:,-1,:]
      rnn_dp_bw = rnn_dp_bw[:,-1,:]
      rnn_cp_bw = rnn_cp_bw[:,-1,:]
      ## concatenate forward and backward: batch_size x 2*embedding_dim
      rnn_dp = torch.cat((rnn_dp_fw, rnn_dp_bw), dim=-1)
      rnn_cp = torch.cat((rnn_cp_fw, rnn_cp_bw), dim=-1)
      
      # Scores
      score_dp = self.fc_dp(self.dropout(rnn_dp))
      score_cp = self.fc_cp(self.dropout(rnn_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []


elif hp.net_variant == 'birnn_time_decay_attention':
  # GRU
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(np.ceil(num_dp_codes**0.25))+1
      self.embed_cp_dim = int(np.ceil(num_cp_codes**0.25))+1

      # Embedding layers
      self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
      self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)

      # GRU layers
      self.gru_dp_fw = GRUExponentialDecay(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim)
      self.gru_cp_fw = GRUExponentialDecay(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim)
      self.gru_dp_bw = GRUExponentialDecay(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim)
      self.gru_cp_bw = GRUExponentialDecay(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim)

      # Attention layers
      self.attention_dp = Attention(embedding_dim=2*self.embed_dp_dim)
      self.attention_cp = Attention(embedding_dim=2*self.embed_cp_dim)
      
      # Fully connected output
      self.fc_dp  = nn.Linear(2*self.embed_dp_dim, 1)
      self.fc_cp  = nn.Linear(2*self.embed_cp_dim, 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Compute time delta
      ## output dim: batch_size x seq_len
      dp_t_delta_fw = abs_time_to_delta(dp_t)
      cp_t_delta_fw = abs_time_to_delta(cp_t)
      dp_t_delta_bw = abs_time_to_delta(torch.flip(dp_t, [1]))
      cp_t_delta_bw = abs_time_to_delta(torch.flip(cp_t, [1]))    
    
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp_fw = self.embed_dp(dp)
      embedded_cp_fw = self.embed_cp(cp)
      embedded_dp_bw = torch.flip(embedded_dp_fw, [1])
      embedded_cp_bw = torch.flip(embedded_cp_fw, [1])
      ## Dropout
      embedded_dp_fw = self.dropout(embedded_dp_fw)
      embedded_cp_fw = self.dropout(embedded_cp_fw)
      embedded_dp_bw = self.dropout(embedded_dp_bw)
      embedded_cp_bw = self.dropout(embedded_cp_bw)
      
      # GRU
      ## output dim rnn:        batch_size x seq_len x embedding_dim
      rnn_dp_fw = self.gru_dp_fw(embedded_dp_fw, dp_t_delta_fw)
      rnn_cp_fw = self.gru_cp_fw(embedded_cp_fw, cp_t_delta_fw)
      rnn_dp_bw = self.gru_dp_bw(embedded_dp_bw, dp_t_delta_bw)
      rnn_cp_bw = self.gru_cp_bw(embedded_cp_bw, cp_t_delta_bw)
      # concatenate forward and backward
      ## output dim: batch_size x seq_len x 2*embedding_dim
      rnn_dp = torch.cat((rnn_dp_fw, torch.flip(rnn_dp_bw, [1])), dim=-1)
      rnn_cp = torch.cat((rnn_cp_fw, torch.flip(rnn_cp_bw, [1])), dim=-1)

      # Attention
      ## output dim: batch_size x 2*embedding_dim
      attended_dp, weights_dp = self.attention_dp(rnn_dp, (dp > 0).float())
      attended_cp, weights_cp = self.attention_cp(rnn_cp, (cp > 0).float())
      
      # Scores
      score_dp = self.fc_dp(self.dropout(attended_dp))
      score_cp = self.fc_cp(self.dropout(attended_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []


elif hp.net_variant == 'attention_concat_time':
  # Attention Only
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(2*np.ceil(num_dp_codes**0.25))
      self.embed_cp_dim = int(2*np.ceil(num_cp_codes**0.25))

      # Embedding layers
      self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
      self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)
      
      # Attention layers
      self.attention_dp = Attention(embedding_dim=self.embed_dp_dim+1) #+1 for the concatenated time
      self.attention_cp = Attention(embedding_dim=self.embed_cp_dim+1)
      
      # Fully connected output
      self.fc_dp  = nn.Linear(self.embed_dp_dim+1, 1)
      self.fc_cp  = nn.Linear(self.embed_cp_dim+1, 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp = self.embed_dp(dp)
      embedded_cp = self.embed_cp(cp)
      
      # Concatate with time
      ## output dim: batch_size x seq_len x (embedding_dim+1)
      concat_dp = torch.cat((embedded_dp, torch.unsqueeze(dp_t, dim=-1)), dim=-1)
      concat_cp = torch.cat((embedded_cp, torch.unsqueeze(cp_t, dim=-1)), dim=-1)
      ## Dropout
      concat_dp = self.dropout(concat_dp)
      concat_cp = self.dropout(concat_cp)
      
      # Attention
      ## output dim: batch_size x (embedding_dim+1)
      attended_dp, weights_dp = self.attention_dp(concat_dp, (dp > 0).float())
      attended_cp, weights_cp = self.attention_cp(concat_cp, (cp > 0).float())
      
      # Scores
      score_dp = self.fc_dp(self.dropout(attended_dp))
      score_cp = self.fc_cp(self.dropout(attended_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []


elif hp.net_variant == 'ode_birnn':
  # Attention Only
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(np.ceil(num_dp_codes**0.25))+1
      self.embed_cp_dim = int(np.ceil(num_cp_codes**0.25))+1

      # Embedding layers
      self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
      self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)
      
      # ODE layers
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.ode_dp = ODENet(self.device, self.embed_dp_dim, self.embed_dp_dim, output_dim=self.embed_dp_dim, augment_dim=0, time_dependent=False, non_linearity='softplus', tol=1e-3, adjoint=True)
      self.ode_cp = ODENet(self.device, self.embed_cp_dim, self.embed_cp_dim, output_dim=self.embed_cp_dim, augment_dim=0, time_dependent=False, non_linearity='softplus', tol=1e-3, adjoint=True)

      # GRU layers
      self.gru_dp_fw = nn.GRU(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim, num_layers=1, batch_first=True)
      self.gru_cp_fw = nn.GRU(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim, num_layers=1, batch_first=True)
      self.gru_dp_bw = nn.GRU(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim, num_layers=1, batch_first=True)
      self.gru_cp_bw = nn.GRU(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim, num_layers=1, batch_first=True)
      
      # Fully connected output
      self.fc_dp  = nn.Linear(2*self.embed_dp_dim, 1)
      self.fc_cp  = nn.Linear(2*self.embed_cp_dim, 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp = self.embed_dp(dp)
      embedded_cp = self.embed_cp(cp)
      
      # ODE
      ## Round times
      dp_t = torch.round(100*dp_t)/100
      cp_t = torch.round(100*cp_t)/100
      
      embedded_dp_long = embedded_dp.view(-1, self.embed_dp_dim)
      dp_t_long = dp_t.view(-1)
      dp_t_long_unique, inverse_indices = torch.unique(dp_t_long, sorted=True, return_inverse=True)
      ode_dp_long = self.ode_dp(embedded_dp_long, dp_t_long_unique)
      ode_dp_long = ode_dp_long[inverse_indices, torch.arange(0, inverse_indices.size(0)), :]
      ode_dp = ode_dp_long.view(dp.size(0), dp.size(1), self.embed_dp_dim)

      embedded_cp_long = embedded_cp.view(-1, self.embed_cp_dim)
      cp_t_long = cp_t.view(-1)
      cp_t_long_unique, inverse_indices = torch.unique(cp_t_long, sorted=True, return_inverse=True)
      ode_cp_long = self.ode_cp(embedded_cp_long, cp_t_long_unique)
      ode_cp_long = ode_cp_long[inverse_indices, torch.arange(0, inverse_indices.size(0)), :]
      ode_cp = ode_cp_long.view(cp.size(0), cp.size(1), self.embed_cp_dim)
      
      ## Dropout
      ode_dp = self.dropout(ode_dp)
      ode_cp = self.dropout(ode_cp)

      # Forward and backward sequences
      ## output dim: batch_size x seq_len x embedding_dim
      ode_dp_fw = ode_dp
      ode_cp_fw = ode_cp
      ode_dp_bw = torch.flip(ode_dp_fw, [1])
      ode_cp_bw = torch.flip(ode_cp_fw, [1])
      
      # GRU
      ## output dim rnn:        batch_size x seq_len x embedding_dim
      ## output dim rnn_hidden: batch_size x 1 x embedding_dim
      rnn_dp_fw, rnn_hidden_dp_fw = self.gru_dp_fw(ode_dp_fw)
      rnn_cp_fw, rnn_hidden_cp_fw = self.gru_cp_fw(ode_cp_fw)
      rnn_dp_bw, rnn_hidden_dp_bw = self.gru_dp_bw(ode_dp_bw)
      rnn_cp_bw, rnn_hidden_cp_bw = self.gru_cp_bw(ode_cp_bw)      
      ## output dim rnn_hidden: batch_size x embedding_dim
      rnn_hidden_dp_fw = rnn_hidden_dp_fw.view(-1, self.embed_dp_dim)
      rnn_hidden_cp_fw = rnn_hidden_cp_fw.view(-1, self.embed_cp_dim)
      rnn_hidden_dp_bw = rnn_hidden_dp_bw.view(-1, self.embed_dp_dim)
      rnn_hidden_cp_bw = rnn_hidden_cp_bw.view(-1, self.embed_cp_dim)
      ## concatenate forward and backward: batch_size x 2*embedding_dim
      rnn_hidden_dp = torch.cat((rnn_hidden_dp_fw, rnn_hidden_dp_bw), dim=-1)
      rnn_hidden_cp = torch.cat((rnn_hidden_cp_fw, rnn_hidden_cp_bw), dim=-1)
      
      # Scores
      score_dp = self.fc_dp(self.dropout(rnn_hidden_dp))
      score_cp = self.fc_cp(self.dropout(rnn_hidden_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []


elif hp.net_variant == 'ode_birnn_attention':
  # Attention Only
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(np.ceil(num_dp_codes**0.25))+1
      self.embed_cp_dim = int(np.ceil(num_cp_codes**0.25))+1

      # Embedding layers
      self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
      self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)
      
      # ODE layers
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.ode_dp = ODENet(self.device, self.embed_dp_dim, self.embed_dp_dim, output_dim=self.embed_dp_dim, augment_dim=0, time_dependent=False, non_linearity='softplus', tol=1e-3, adjoint=True)
      self.ode_cp = ODENet(self.device, self.embed_cp_dim, self.embed_cp_dim, output_dim=self.embed_cp_dim, augment_dim=0, time_dependent=False, non_linearity='softplus', tol=1e-3, adjoint=True)

      # GRU layers
      self.gru_dp_fw = nn.GRU(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim, num_layers=1, batch_first=True)
      self.gru_cp_fw = nn.GRU(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim, num_layers=1, batch_first=True)
      self.gru_dp_bw = nn.GRU(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim, num_layers=1, batch_first=True)
      self.gru_cp_bw = nn.GRU(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim, num_layers=1, batch_first=True)

      # Attention layers
      self.attention_dp = Attention(embedding_dim=2*self.embed_dp_dim)
      self.attention_cp = Attention(embedding_dim=2*self.embed_cp_dim)
      
      # Fully connected output
      self.fc_dp  = nn.Linear(2*self.embed_dp_dim, 1)
      self.fc_cp  = nn.Linear(2*self.embed_cp_dim, 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp = self.embed_dp(dp)
      embedded_cp = self.embed_cp(cp)
      
      # ODE
      ## Round times
      dp_t = torch.round(100*dp_t)/100
      cp_t = torch.round(100*cp_t)/100
      
      embedded_dp_long = embedded_dp.view(-1, self.embed_dp_dim)
      dp_t_long = dp_t.view(-1)
      dp_t_long_unique, inverse_indices = torch.unique(dp_t_long, sorted=True, return_inverse=True)
      ode_dp_long = self.ode_dp(embedded_dp_long, dp_t_long_unique)
      ode_dp_long = ode_dp_long[inverse_indices, torch.arange(0, inverse_indices.size(0)), :]
      ode_dp = ode_dp_long.view(dp.size(0), dp.size(1), self.embed_dp_dim)

      embedded_cp_long = embedded_cp.view(-1, self.embed_cp_dim)
      cp_t_long = cp_t.view(-1)
      cp_t_long_unique, inverse_indices = torch.unique(cp_t_long, sorted=True, return_inverse=True)
      ode_cp_long = self.ode_cp(embedded_cp_long, cp_t_long_unique)
      ode_cp_long = ode_cp_long[inverse_indices, torch.arange(0, inverse_indices.size(0)), :]
      ode_cp = ode_cp_long.view(cp.size(0), cp.size(1), self.embed_cp_dim)
      
      ## Dropout
      ode_dp = self.dropout(ode_dp)
      ode_cp = self.dropout(ode_cp)

      # Forward and backward sequences
      ## output dim: batch_size x seq_len x embedding_dim
      ode_dp_fw = ode_dp
      ode_cp_fw = ode_cp
      ode_dp_bw = torch.flip(ode_dp_fw, [1])
      ode_cp_bw = torch.flip(ode_cp_fw, [1])
      
      # GRU
      ## output dim rnn:        batch_size x seq_len x embedding_dim
      ## output dim rnn_hidden: batch_size x 1 x embedding_dim
      rnn_dp_fw, rnn_hidden_dp_fw = self.gru_dp_fw(ode_dp_fw)
      rnn_cp_fw, rnn_hidden_cp_fw = self.gru_cp_fw(ode_cp_fw)
      rnn_dp_bw, rnn_hidden_dp_bw = self.gru_dp_bw(ode_dp_bw)
      rnn_cp_bw, rnn_hidden_cp_bw = self.gru_cp_bw(ode_cp_bw)      
      # concatenate forward and backward
      ## output dim: batch_size x seq_len x 2*embedding_dim
      rnn_dp = torch.cat((rnn_dp_fw, torch.flip(rnn_dp_bw, [1])), dim=-1)
      rnn_cp = torch.cat((rnn_cp_fw, torch.flip(rnn_cp_bw, [1])), dim=-1)

      # Attention
      ## output dim: batch_size x 2*embedding_dim
      attended_dp, weights_dp = self.attention_dp(rnn_dp, (dp > 0).float())
      attended_cp, weights_cp = self.attention_cp(rnn_cp, (cp > 0).float())
      
      # Scores
      score_dp = self.fc_dp(self.dropout(attended_dp))
      score_cp = self.fc_cp(self.dropout(attended_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []


elif hp.net_variant == 'ode_attention':
  # Attention Only
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(2*np.ceil(num_dp_codes**0.25))+1
      self.embed_cp_dim = int(2*np.ceil(num_cp_codes**0.25))+1

      # Embedding layers
      self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
      self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)
      
      # ODE layers
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.ode_dp = ODENet(self.device, self.embed_dp_dim, self.embed_dp_dim, output_dim=self.embed_dp_dim, augment_dim=0, time_dependent=False, non_linearity='softplus', tol=1e-3, adjoint=True)
      self.ode_cp = ODENet(self.device, self.embed_cp_dim, self.embed_cp_dim, output_dim=self.embed_cp_dim, augment_dim=0, time_dependent=False, non_linearity='softplus', tol=1e-3, adjoint=True)
      
      # Attention layers
      self.attention_dp = Attention(embedding_dim=self.embed_dp_dim)
      self.attention_cp = Attention(embedding_dim=self.embed_cp_dim)
      
      # Fully connected output
      self.fc_dp  = nn.Linear(self.embed_dp_dim, 1)
      self.fc_cp  = nn.Linear(self.embed_cp_dim, 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp = self.embed_dp(dp)
      embedded_cp = self.embed_cp(cp)
      
      # ODE
      ## Round times
      dp_t = torch.round(100*dp_t)/100
      cp_t = torch.round(100*cp_t)/100
      
      embedded_dp_long = embedded_dp.view(-1, self.embed_dp_dim)
      dp_t_long = dp_t.view(-1)
      dp_t_long_unique, inverse_indices = torch.unique(dp_t_long, sorted=True, return_inverse=True)
      ode_dp_long = self.ode_dp(embedded_dp_long, dp_t_long_unique)
      ode_dp_long = ode_dp_long[inverse_indices, torch.arange(0, inverse_indices.size(0)), :]
      ode_dp = ode_dp_long.view(dp.size(0), dp.size(1), self.embed_dp_dim)

      embedded_cp_long = embedded_cp.view(-1, self.embed_cp_dim)
      cp_t_long = cp_t.view(-1)
      cp_t_long_unique, inverse_indices = torch.unique(cp_t_long, sorted=True, return_inverse=True)
      ode_cp_long = self.ode_cp(embedded_cp_long, cp_t_long_unique)
      ode_cp_long = ode_cp_long[inverse_indices, torch.arange(0, inverse_indices.size(0)), :]
      ode_cp = ode_cp_long.view(cp.size(0), cp.size(1), self.embed_cp_dim)

      ## Dropout
      ode_dp = self.dropout(ode_dp)
      ode_cp = self.dropout(ode_cp)
      
      # Attention
      ## output dim: batch_size x (embedding_dim+1)
      attended_dp, weights_dp = self.attention_dp(ode_dp, (dp > 0).float())
      attended_cp, weights_cp = self.attention_cp(ode_cp, (cp > 0).float())
      
      # Scores
      score_dp = self.fc_dp(self.dropout(attended_dp))
      score_cp = self.fc_cp(self.dropout(attended_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []


elif hp.net_variant == 'birnn_ode_decay':
  # GRU
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(np.ceil(num_dp_codes**0.25))+1
      self.embed_cp_dim = int(np.ceil(num_cp_codes**0.25))+1

      # Embedding layers
      self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
      self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)

      # GRU layers
      self.gru_dp_fw = GRUOdeDecay(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim)
      self.gru_cp_fw = GRUOdeDecay(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim)
      self.gru_dp_bw = GRUOdeDecay(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim)
      self.gru_cp_bw = GRUOdeDecay(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim)
      
      # Fully connected output
      self.fc_dp  = nn.Linear(2*self.embed_dp_dim, 1)
      self.fc_cp  = nn.Linear(2*self.embed_cp_dim, 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Compute time delta
      ## output dim: batch_size x seq_len
      dp_t_delta_fw = abs_time_to_delta(dp_t)
      cp_t_delta_fw = abs_time_to_delta(cp_t)
      ## Round
      dp_t_delta_fw = torch.round(100*dp_t_delta_fw)/100
      cp_t_delta_fw = torch.round(100*cp_t_delta_fw)/100            
      dp_t_delta_bw = abs_time_to_delta(torch.flip(dp_t, [1]))
      cp_t_delta_bw = abs_time_to_delta(torch.flip(cp_t, [1]))    
    
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp_fw = self.embed_dp(dp)
      embedded_cp_fw = self.embed_cp(cp)
      embedded_dp_bw = torch.flip(embedded_dp_fw, [1])
      embedded_cp_bw = torch.flip(embedded_cp_fw, [1])
      ## Dropout
      embedded_dp_fw = self.dropout(embedded_dp_fw)
      embedded_cp_fw = self.dropout(embedded_cp_fw)
      embedded_dp_bw = self.dropout(embedded_dp_bw)
      embedded_cp_bw = self.dropout(embedded_cp_bw)
      
      # GRU
      ## output dim rnn:        batch_size x seq_len x embedding_dim
      rnn_dp_fw = self.gru_dp_fw(embedded_dp_fw, dp_t_delta_fw)
      rnn_cp_fw = self.gru_cp_fw(embedded_cp_fw, cp_t_delta_fw)
      rnn_dp_bw = self.gru_dp_bw(embedded_dp_bw, dp_t_delta_bw)
      rnn_cp_bw = self.gru_cp_bw(embedded_cp_bw, cp_t_delta_bw)      
      ## output dim rnn_hidden: batch_size x embedding_dim
      rnn_dp_fw = rnn_dp_fw[:,-1,:]
      rnn_cp_fw = rnn_cp_fw[:,-1,:]
      rnn_dp_bw = rnn_dp_bw[:,-1,:]
      rnn_cp_bw = rnn_cp_bw[:,-1,:]
      ## concatenate forward and backward: batch_size x 2*embedding_dim
      rnn_dp = torch.cat((rnn_dp_fw, rnn_dp_bw), dim=-1)
      rnn_cp = torch.cat((rnn_cp_fw, rnn_cp_bw), dim=-1)
      
      # Scores
      score_dp = self.fc_dp(self.dropout(rnn_dp))
      score_cp = self.fc_cp(self.dropout(rnn_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []      
      

elif hp.net_variant == 'birnn_ode_decay_attention':
  # GRU
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(np.ceil(num_dp_codes**0.25))+1
      self.embed_cp_dim = int(np.ceil(num_cp_codes**0.25))+1

      # Embedding layers
      self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
      self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)

      # GRU layers
      self.gru_dp_fw = GRUOdeDecay(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim)
      self.gru_cp_fw = GRUOdeDecay(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim)
      self.gru_dp_bw = GRUOdeDecay(input_size=self.embed_dp_dim, hidden_size=self.embed_dp_dim)
      self.gru_cp_bw = GRUOdeDecay(input_size=self.embed_cp_dim, hidden_size=self.embed_cp_dim)

      # Attention layers
      self.attention_dp = Attention(embedding_dim=2*self.embed_dp_dim)
      self.attention_cp = Attention(embedding_dim=2*self.embed_cp_dim)
      
      # Fully connected output
      self.fc_dp  = nn.Linear(2*self.embed_dp_dim, 1)
      self.fc_cp  = nn.Linear(2*self.embed_cp_dim, 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Compute time delta
      ## output dim: batch_size x seq_len
      dp_t_delta_fw = abs_time_to_delta(dp_t)
      cp_t_delta_fw = abs_time_to_delta(cp_t)
      ## Round
      dp_t_delta_fw = torch.round(100*dp_t_delta_fw)/100
      cp_t_delta_fw = torch.round(100*cp_t_delta_fw)/100      
      dp_t_delta_bw = abs_time_to_delta(torch.flip(dp_t, [1]))
      cp_t_delta_bw = abs_time_to_delta(torch.flip(cp_t, [1]))    
    
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp_fw = self.embed_dp(dp)
      embedded_cp_fw = self.embed_cp(cp)
      embedded_dp_bw = torch.flip(embedded_dp_fw, [1])
      embedded_cp_bw = torch.flip(embedded_cp_fw, [1])
      ## Dropout
      embedded_dp_fw = self.dropout(embedded_dp_fw)
      embedded_cp_fw = self.dropout(embedded_cp_fw)
      embedded_dp_bw = self.dropout(embedded_dp_bw)
      embedded_cp_bw = self.dropout(embedded_cp_bw)
      
      # GRU
      ## output dim rnn:        batch_size x seq_len x embedding_dim
      rnn_dp_fw = self.gru_dp_fw(embedded_dp_fw, dp_t_delta_fw)
      rnn_cp_fw = self.gru_cp_fw(embedded_cp_fw, cp_t_delta_fw)
      rnn_dp_bw = self.gru_dp_bw(embedded_dp_bw, dp_t_delta_bw)
      rnn_cp_bw = self.gru_cp_bw(embedded_cp_bw, cp_t_delta_bw)
      # concatenate forward and backward
      ## output dim: batch_size x seq_len x 2*embedding_dim
      rnn_dp = torch.cat((rnn_dp_fw, torch.flip(rnn_dp_bw, [1])), dim=-1)
      rnn_cp = torch.cat((rnn_cp_fw, torch.flip(rnn_cp_bw, [1])), dim=-1)

      # Attention
      ## output dim: batch_size x 2*embedding_dim
      attended_dp, weights_dp = self.attention_dp(rnn_dp, (dp > 0).float())
      attended_cp, weights_cp = self.attention_cp(rnn_cp, (cp > 0).float())
      
      # Scores
      score_dp = self.fc_dp(self.dropout(attended_dp))
      score_cp = self.fc_cp(self.dropout(attended_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []


elif hp.net_variant == 'mce_attention':
  # Attention Only
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(2*np.ceil(num_dp_codes**0.25))
      self.embed_cp_dim = int(2*np.ceil(num_cp_codes**0.25))

      # Precomputed embedding weights
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.emb_weight_dp = torch.Tensor(np.load(hp.data_dir + 'emb_weight_dp_13.npy')).to(self.device)
      self.emb_weight_cp = torch.Tensor(np.load(hp.data_dir + 'emb_weight_cp_11.npy')).to(self.device)
      
      # Attention layers
      self.attention_dp = Attention(embedding_dim=self.embed_dp_dim+1) #+1 for the concatenated time
      self.attention_cp = Attention(embedding_dim=self.embed_cp_dim+1)
      
      # Fully connected output
      self.fc_dp  = nn.Linear(self.embed_dp_dim+1, 1)
      self.fc_cp  = nn.Linear(self.embed_cp_dim+1, 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp = F.embedding(dp, self.emb_weight_dp, padding_idx=0)
      embedded_cp = F.embedding(cp, self.emb_weight_cp, padding_idx=0)
      ## Dropout
      embedded_dp = self.dropout(embedded_dp)
      embedded_cp = self.dropout(embedded_cp)
      
      # Attention
      ## output dim: batch_size x (embedding_dim+1)
      attended_dp, weights_dp = self.attention_dp(embedded_dp, (dp > 0).float())
      attended_cp, weights_cp = self.attention_cp(embedded_cp, (cp > 0).float())
      
      # Scores
      score_dp = self.fc_dp(self.dropout(attended_dp))
      score_cp = self.fc_cp(self.dropout(attended_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []


if hp.net_variant == 'mce_birnn':
  # GRU
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(np.ceil(num_dp_codes**0.25))
      self.embed_cp_dim = int(np.ceil(num_cp_codes**0.25))

      # Precomputed embedding weights
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.emb_weight_dp = torch.Tensor(np.load(hp.data_dir + 'emb_weight_dp_7.npy')).to(self.device)
      self.emb_weight_cp = torch.Tensor(np.load(hp.data_dir + 'emb_weight_cp_6.npy')).to(self.device)

      # GRU layers
      self.gru_dp_fw = nn.GRU(input_size=self.embed_dp_dim+1, hidden_size=self.embed_dp_dim+1, num_layers=1, batch_first=True)
      self.gru_cp_fw = nn.GRU(input_size=self.embed_cp_dim+1, hidden_size=self.embed_cp_dim+1, num_layers=1, batch_first=True)
      self.gru_dp_bw = nn.GRU(input_size=self.embed_dp_dim+1, hidden_size=self.embed_dp_dim+1, num_layers=1, batch_first=True)
      self.gru_cp_bw = nn.GRU(input_size=self.embed_cp_dim+1, hidden_size=self.embed_cp_dim+1, num_layers=1, batch_first=True)
      
      # Fully connected output
      self.fc_dp  = nn.Linear(2*(self.embed_dp_dim+1), 1)
      self.fc_cp  = nn.Linear(2*(self.embed_cp_dim+1), 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp_fw = F.embedding(dp, self.emb_weight_dp, padding_idx=0)
      embedded_cp_fw = F.embedding(cp, self.emb_weight_cp, padding_idx=0)
      embedded_dp_bw = torch.flip(embedded_dp_fw, [1])
      embedded_cp_bw = torch.flip(embedded_cp_fw, [1])
      ## Dropout
      embedded_dp_fw = self.dropout(embedded_dp_fw)
      embedded_cp_fw = self.dropout(embedded_cp_fw)
      embedded_dp_bw = self.dropout(embedded_dp_bw)
      embedded_cp_bw = self.dropout(embedded_cp_bw)
      
      # GRU
      ## output dim rnn:        batch_size x seq_len x (embedding_dim+1)
      ## output dim rnn_hidden: batch_size x 1 x (embedding_dim+1)
      rnn_dp_fw, rnn_hidden_dp_fw = self.gru_dp_fw(embedded_dp_fw)
      rnn_cp_fw, rnn_hidden_cp_fw = self.gru_cp_fw(embedded_cp_fw)
      rnn_dp_bw, rnn_hidden_dp_bw = self.gru_dp_bw(embedded_dp_bw)
      rnn_cp_bw, rnn_hidden_cp_bw = self.gru_cp_bw(embedded_cp_bw)      
      ## output dim rnn_hidden: batch_size x (embedding_dim+1)
      rnn_hidden_dp_fw = rnn_hidden_dp_fw.view(-1, self.embed_dp_dim+1)
      rnn_hidden_cp_fw = rnn_hidden_cp_fw.view(-1, self.embed_cp_dim+1)
      rnn_hidden_dp_bw = rnn_hidden_dp_bw.view(-1, self.embed_dp_dim+1)
      rnn_hidden_cp_bw = rnn_hidden_cp_bw.view(-1, self.embed_cp_dim+1)
      ## concatenate forward and backward: batch_size x 2*(embedding_dim+1)
      rnn_hidden_dp = torch.cat((rnn_hidden_dp_fw, rnn_hidden_dp_bw), dim=-1)
      rnn_hidden_cp = torch.cat((rnn_hidden_cp_fw, rnn_hidden_cp_bw), dim=-1)
      
      # Scores
      score_dp = self.fc_dp(self.dropout(rnn_hidden_dp))
      score_cp = self.fc_cp(self.dropout(rnn_hidden_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []


elif hp.net_variant == 'mce_birnn_attention':
  # GRU
  class Net(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
      super(Net, self).__init__()
      
      # Embedding dimensions
      self.embed_dp_dim = int(np.ceil(num_dp_codes**0.25))
      self.embed_cp_dim = int(np.ceil(num_cp_codes**0.25))

      # Precomputed embedding weights
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.emb_weight_dp = torch.Tensor(np.load(hp.data_dir + 'emb_weight_dp_7.npy')).to(self.device)
      self.emb_weight_cp = torch.Tensor(np.load(hp.data_dir + 'emb_weight_cp_6.npy')).to(self.device)

      # GRU layers
      self.gru_dp_fw = nn.GRU(input_size=self.embed_dp_dim+1, hidden_size=self.embed_dp_dim+1, num_layers=1, batch_first=True)
      self.gru_cp_fw = nn.GRU(input_size=self.embed_cp_dim+1, hidden_size=self.embed_cp_dim+1, num_layers=1, batch_first=True)
      self.gru_dp_bw = nn.GRU(input_size=self.embed_dp_dim+1, hidden_size=self.embed_dp_dim+1, num_layers=1, batch_first=True)
      self.gru_cp_bw = nn.GRU(input_size=self.embed_cp_dim+1, hidden_size=self.embed_cp_dim+1, num_layers=1, batch_first=True)

      # Attention layers
      self.attention_dp = Attention(embedding_dim=2*(self.embed_dp_dim+1)) #+1 for the concatenated time
      self.attention_cp = Attention(embedding_dim=2*(self.embed_cp_dim+1))
            
      # Fully connected output
      self.fc_dp  = nn.Linear(2*(self.embed_dp_dim+1), 1)
      self.fc_cp  = nn.Linear(2*(self.embed_cp_dim+1), 1)
      self.fc_all = nn.Linear(num_static + 2, 1)
      
      # Others
      self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
      # Embedding
      ## output dim: batch_size x seq_len x embedding_dim
      embedded_dp_fw = F.embedding(dp, self.emb_weight_dp, padding_idx=0)
      embedded_cp_fw = F.embedding(cp, self.emb_weight_cp, padding_idx=0)
      embedded_dp_bw = torch.flip(embedded_dp_fw, [1])
      embedded_cp_bw = torch.flip(embedded_cp_fw, [1])
      ## Dropout
      embedded_dp_fw = self.dropout(embedded_dp_fw)
      embedded_cp_fw = self.dropout(embedded_cp_fw)
      embedded_dp_bw = self.dropout(embedded_dp_bw)
      embedded_cp_bw = self.dropout(embedded_cp_bw)
      
      # GRU
      ## output dim rnn:        batch_size x seq_len x (embedding_dim+1)
      ## output dim rnn_hidden: batch_size x 1 x (embedding_dim+1)
      rnn_dp_fw, rnn_hidden_dp_fw = self.gru_dp_fw(embedded_dp_fw)
      rnn_cp_fw, rnn_hidden_cp_fw = self.gru_cp_fw(embedded_cp_fw)
      rnn_dp_bw, rnn_hidden_dp_bw = self.gru_dp_bw(embedded_dp_bw)
      rnn_cp_bw, rnn_hidden_cp_bw = self.gru_cp_bw(embedded_cp_bw)   
      # concatenate forward and backward
      ## output dim: batch_size x seq_len x 2*(embedding_dim+1)
      rnn_dp = torch.cat((rnn_dp_fw, torch.flip(rnn_dp_bw, [1])), dim=-1)
      rnn_cp = torch.cat((rnn_cp_fw, torch.flip(rnn_cp_bw, [1])), dim=-1)

      # Attention
      ## output dim: batch_size x 2*(embedding_dim+1)
      attended_dp, weights_dp = self.attention_dp(rnn_dp, (dp > 0).float())
      attended_cp, weights_cp = self.attention_cp(rnn_cp, (cp > 0).float())
      
      # Scores
      score_dp = self.fc_dp(self.dropout(attended_dp))
      score_cp = self.fc_cp(self.dropout(attended_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []
      