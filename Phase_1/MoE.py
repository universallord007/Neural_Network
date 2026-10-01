import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


n_embd = 384
num_experts = 4

# This is the simple MLP 
class Expert(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc  = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# This is the Mixture of Experts which is the main core
class MoE(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.top_k = 2
        self.router = nn.Linear(config.n_embd, config.num_experts)
        self.experts = nn.ModuleList(
            [Expert(config) for _ in range(config.num_experts)]
        )

    def forward(self,x):
        router_logits = self.router(x)
        probs = F.softmax(router_logits, dim= -1)
        top_probs , top_indices = torch.topk(probs, k = 2 , dim = -1)
        # Here we are normalizing the probs
        top_probs = top_probs/top_probs.sum(dim =-1, keepdims= True)
        # load balancing code 
        # Then the common simple auxiliary loss is:
        # L_balance = num_experts × Σ(fᵢ × Pᵢ)
        # Where fi is the fraction of tokens routed to expert i and the pi is the average routing probability of expert i
        B,T,C = x.shape
        balance_loss = 0
        for expert_id in range(len(self.experts)):
            count = (top_indices == expert_id).sum()
            fraction = count/(B * T * self.top_k)
            avg_probs = probs[:,:,expert_id].mean()
            balance_loss += fraction*avg_probs
            # print(f"Expert {expert_id}: {count} selections")
            # print(f"The balance loss is {balance_loss}")
            # print(f'The expert {expert_id} has the tokens fraction of {fraction}')
        L_balance = len(self.experts)*balance_loss
        # print(f"The L_balance of expert {expert_id} is {L_balance}")


        output = torch.zeros_like(x)
        for k in range(self.top_k) :
            index = top_indices[:,:,k]
            for expert_id in range(len(self.experts)):
                mask = (index == expert_id)
                tokens = x[mask]
                weights = top_probs[:,:,k][mask]
                expert_output = self.experts[expert_id](tokens)
                weighted_output = expert_output * weights.unsqueeze(-1)
                output[mask] += weighted_output
        return output,L_balance


x = torch.randn(3,5,384)

@dataclass
class Config:
    n_embd : int = 384
    num_experts : int = 4

config = Config()
moe = MoE(config)
x = torch.randn(3,5,384)
output,L_bal = moe(x)
# print(f"The shape of output is {output.shape}")
# print(f"The loss balance is {L_bal}")


                      