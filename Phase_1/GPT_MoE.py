from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import time 



@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    num_experts: int = 4


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Key, Query and Value projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        # Calculate query, key and value
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)



        # # Attention scores
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # Apply causal mask
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0,float("-inf"))
        # # Softmax
        # att = F.softmax(att, dim=-1)
        # # Attention output
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal = True)

        # Recombine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)

        return y


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
        # print("Forward is being called !!!")
        router_logits = self.router(x)
        probs = F.softmax(router_logits, dim= -1)
        top_probs , top_indices = torch.topk(probs, k = 2 , dim = -1)
        # Here we are normalizing the probs
        top_probs = top_probs/top_probs.sum(dim =-1, keepdims= True)
        # load balancing code 
        # Then the common simple auxiliary loss is:
        # L_balance = num_experts × Σ(fᵢ × Pᵢ)
        # Where fi is the fraction of tokens routed to expert i and the pi is the average routing probability of expert i
        print(top_probs.sum(dim=-1))
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


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MoE(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        moe_output, balance_loss = self.moe(self.ln_2(x))
        x = x + moe_output
        return x,balance_loss


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.vocab_size,
                    config.n_embd
                ),

                wpe=nn.Embedding(
                    config.block_size,
                    config.n_embd
                ),

                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),

                ln_f=nn.LayerNorm(config.n_embd)
            )
        )

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2*self.config.n_layer)**-0.5
            module.weight.data.normal_(mean=0.0, std=std)

            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self,idx,targets=None):
      B,T = idx.size()
      pos = torch.arange(0,T, dtype=torch.long, device = idx.device)
      pos_emb = self.transformer.wpe(pos)
      tok_emb = self.transformer.wte(idx)
      x = tok_emb + pos_emb
      total_loss = 0
      for block in self.transformer.h:
        x,balance_loss = block(x)
        total_loss += balance_loss
      x = self.transformer.ln_f(x)
      logits= self.lm_head(x)
      loss = None
      if targets is not None:
        loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        total_loss = loss + 0.1*balance_loss
      else:
        total_loss = None
      return logits,total_loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from HuggingFace."""

        assert model_type in {
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl"
        }

        from transformers import GPT2LMHeadModel

        print(
            "Loading weights from pretrained GPT: %s"
            % model_type
        )

        # n_layer, n_head and n_embd depend on model type
        config_args = {
            "gpt2": dict(
                n_layer=12,
                n_head=12,
                n_embd=768
            ),

            "gpt2-medium": dict(
                n_layer=24,
                n_head=16,
                n_embd=1024
            ),

            "gpt2-large": dict(
                n_layer=36,
                n_head=20,
                n_embd=1280
            ),

            "gpt2-xl": dict(
                n_layer=48,
                n_head=25,
                n_embd=1600
            ),
        }[model_type]

        # GPT-2 constants
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        # Create our GPT model
        config = GPTConfig(**config_args)
        model = cls(config)

        # Our model state dict
        sd = model.state_dict()

        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys
            if not k.endswith(".attn.bias")
        ]

        # Load HuggingFace GPT-2
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)

        sd_hf = model_hf.state_dict()

        # HuggingFace state dict keys
        sd_keys_hf = sd_hf.keys()

        sd_keys_hf = [
            k for k in sd_keys_hf
            if not k.endswith(".attn.masked_bias")
        ]

        sd_keys_hf = [
            k for k in sd_keys_hf
            if not k.endswith(".attn.bias")
        ]

        # These weights need transposing because HF GPT-2
        # uses Conv1D instead of nn.Linear
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "moe.c_fc.weight",
            "moe.c_proj.weight"
        ]

        # Ensure parameters match
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: "
            f"{len(sd_keys_hf)} != {len(sd_keys)}"
        )

        # Copy HuggingFace weights into our model
        for k in sd_keys_hf:

            if any(k.endswith(w) for w in transposed):

                # Conv1D weights must be transposed
                assert sd_hf[k].shape[::-1] == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                # Direct copy
                assert sd_hf[k].shape == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



device ='cpu'
if torch.cuda.is_available():
  device ='cuda'
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
  device ='mps'
print(f"using device : {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.cuda.manual_seed(1337)
# elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
#   torch.backends.mps.manual_seed(1337)



num_return_sequences = 5
max_length = 30
# model = GPT.from_pretrained("gpt2")
import tiktoken
import torch

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.encoding_for_model('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

train_loader = DataLoaderLite(B=4,T=32)

torch.set_float32_matmul_precision("high")


model = GPT(GPTConfig(vocab_size = 50304))
model.eval()
model.to(device)
model = torch.compile(model)


max_lr = 6e-4
min_lr = max_lr*0.1
warmup_steps = 10
max_steps = 50
import math
import inspect
import torch


# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas =(0.9,0.95), eps = 1e-8)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
for step in range(100):
  t0 = time.time()
  x,y = train_loader.next_batch()
  x = x.to(device)
  y = y.to(device)
  optimizer.zero_grad(set_to_none=True)
  with torch.autocast(device_type="cuda", dtype = torch.bfloat16):
    logits,loss = model(x,y)
  loss.backward()
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  optimizer.step()
  torch.cuda.synchronize()
  t1 = time.time()
  # Printing the time in miliseconds
  d = (t1 - t0)*1000
  tokens_per_sec = (train_loader.B * train_loader.T) / (d*1000)
  print(f"Loss | {loss.item()} time diff | {d:.2f}ms norm | {norm:.4f} tokens/sec | {tokens_per_sec}")




# This is the evaluation part
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I am a language model")
tokens = torch.tensor(tokens, dtype = torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x = tokens.to('cuda')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
  model.eval()
  with torch.no_grad():
    #Taking the predictions
    logits,loss = model(x)
    #Taking the logits at last position
    logits = logits[:,-1,:]
    #Getting the probabilities
    probs = F.softmax(logits, dim =-1)
    #Getting the top 50 probs of the last dimension
    topk_probs , topk_indices = torch.topk(probs, 50 , dim=-1)
    ix = torch.multinomial(topk_probs,1)
    xcol = torch.gather(topk_indices,-1,ix)
    x = torch.cat((x,xcol),dim=1)

for i in range(num_return_sequences):
  tokens = x[i,:max_length].tolist()
  decoded = enc.decode(tokens)
  print("<",decoded)



