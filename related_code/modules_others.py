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


class AttentionImportanceDecay(Attention):
  """
  Dot-product attention module with code-specific importance decay.
  
  Args:
    inputs: A `Tensor` with embeddings in the last dimension.
    decays: A `Tensor`. Dimensions are the same as inputs but the last (embedding) dimension is one. 
      It represents the exponential decay to be applied to the weight of each code depending on time of recording.
    times: A `Tensor` with the same shape as inputs containing the recorded times (but no embedding dimension).
    mask: A `Tensor`. Dimensions are the same as inputs but without the embedding dimension.
      Values are 0 for 0-padding in the input and 1 elsewhere.

  Returns:
    outputs: The input `Tensor` whose embeddings in the last dimension have undergone a weighted average.
      The second-last dimension of the `Tensor` is removed.
    attention_weights: weights given to each embedding.
  """
  def forward(self, inputs, decays, times, mask):
    # Hidden representation of embeddings (no change in dimensions)
    hidden = torch.tanh(self.linear_hidden(inputs))
    # Compute weight of each embedding
    importance = torch.sum(hidden * self.context, dim=-1)
    # Adjust importance according to how long ago the embedding was recorded
    importance = importance - torch.abs(torch.squeeze(decays)) * times
    importance = importance.masked_fill(mask == 0, -1e9)
    # Softmax so that weights sum up to one
    attention_weights = F.softmax(importance, dim=-1)    
    # Weighted sum of embeddings
    weighted_projection = inputs * torch.unsqueeze(attention_weights, dim=-1)
    # Output
    outputs = torch.sum(weighted_projection, dim=-2)
    return outputs, attention_weights 


class SelfAttention(torch.nn.Module):
  """
  Self attention module from "Attention is all you need".
  
  Args:
    inputs: A `Tensor` with embeddings in the last dimension.
    mask: A `Tensor`. Dimensions are the same as inputs but without the embedding dimension.
      Values are 0 for 0-padding in the input and 1 elsewhere.

  Returns:
    outputs: The input `Tensor` whose (transformed) embeddings in the last dimension have undergone a weighted average.
      The second-last dimension of the `Tensor` is removed.
    attention_weights: weights given to each embedding.
  """
  def __init__(self, embedding_dim):
    super(SelfAttention, self).__init__()
    self.linear_keys = nn.Linear(embedding_dim, embedding_dim)
    self.linear_queries = nn.Linear(embedding_dim, embedding_dim)
    self.linear_values = nn.Linear(embedding_dim, embedding_dim)
    self.norm_factor = np.sqrt(embedding_dim)

  def forward(self, inputs, mask):
    # Keys/Queries/Values
    keys = self.linear_keys(inputs)
    queries = self.linear_queries(inputs)
    values = self.linear_values(inputs)
    # Compute weight of each embedding
    importance = torch.matmul(queries, torch.transpose(keys, -2, -1)) / self.norm_factor
    # Softmax so that weights sum up to one
    importance = importance.masked_fill(torch.unsqueeze(mask, dim=-2) == 0, -1e9)
    attention_weights = F.softmax(importance, dim=-1)    
    # Weighted sum of embeddings
    outputs = torch.matmul(attention_weights, values)
    return outputs


if hp.net_variant == 'attention_importance_decay':
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
      self.decay_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=1, padding_idx=0)
      self.decay_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=1, padding_idx=0)      
      
      # Attention layers
      self.attention_dp = AttentionImportanceDecay(embedding_dim=self.embed_dp_dim)
      self.attention_cp = AttentionImportanceDecay(embedding_dim=self.embed_cp_dim)
      
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
      decays_dp   = self.decay_dp(dp)
      decays_cp   = self.decay_cp(cp)      
            
      # Attention
      ## output dim: batch_size x embedding_dim
      attended_dp, weights_dp = self.attention_dp(embedded_dp, decays_dp, dp_t, (dp > 0).float())
      attended_cp, weights_cp = self.attention_cp(embedded_cp, decays_cp, cp_t, (cp > 0).float())
      
      # Scores
      score_dp = self.fc_dp(self.dropout(attended_dp))
      score_cp = self.fc_cp(self.dropout(attended_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []
      
      
elif hp.net_variant == 'transformer_concat_time':
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
      
      # Self attention layers
      self.self_attention_dp = SelfAttention(embedding_dim=self.embed_dp_dim+1) #+1 for the concatenated time
      self.self_attention_cp = SelfAttention(embedding_dim=self.embed_cp_dim+1)
      
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

      # Self Attention
      ## output dim: batch_size x seq_len x (embedding_dim+1)
      sa_dp = self.self_attention_dp(concat_dp, (dp > 0).float())
      sa_cp = self.self_attention_cp(concat_cp, (cp > 0).float())
      
      # Attention
      ## output dim: batch_size x (embedding_dim+1)
      attended_dp, weights_dp = self.attention_dp(sa_dp, (dp > 0).float())
      attended_cp, weights_cp = self.attention_cp(sa_cp, (cp > 0).float())
      
      # Scores
      score_dp = self.fc_dp(self.dropout(attended_dp))
      score_cp = self.fc_cp(self.dropout(attended_cp))

      # Concatenate to variable collection
      all = torch.cat((stat, score_dp, score_cp), dim=1)
      
      # Final linear projection
      out = self.fc_all(self.dropout(all)).squeeze()

      return out, []