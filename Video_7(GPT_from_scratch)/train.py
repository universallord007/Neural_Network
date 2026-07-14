import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


batch_size = 32     # how many independent sequences will we process in parallel?
block_size = 8      # what is the maximum context length for predictions?
max_iters = 3000    # total training iterations
eval_interval = 300 # how often do we check train/val loss?
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200    # how many batches to average out for a clean loss estimate
n_embd = 32


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out





# This is a Bigram Language Model 
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx





with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

# print(len(text))
# print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("".join(chars))
# print(vocab_size)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
# This is an encoder which will convert the text into list of integars
encode = lambda s: [stoi[c] for c in s]
# This is a decoder which will convert the integers back into the required text
decode = lambda l : ''.join([itos[i] for i in l])

# print(encode("Hi there"))
# print(decode(encode("Hi there")))

data = torch.tensor(encode(text), dtype = torch.long)
# print(data[:1000])
# print(data.shape, data.dtype)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# This was just the tutorial
# block_size = 8
# x = train_data[:block_size]
# y = train_data[1:block_size+1]

# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"When input is {context} the target is {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)

# # print('----')

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target: {target}")


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
# print(decode(m.generate(idx=torch.zeros((1,1), dtype = torch.long),max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.Adam(m.parameters(), lr= 1e-3)
batch_size = 32

# for _ in range(10000):
#     # Getting a batch for training
#     xb, yb = get_batch('train')
#     logits,loss = m(xb,yb)
#     optimizer.zero_grad(set_to_none = True)
#     loss.backward()
#     optimizer.step()

B,T,C = 4,8,2
x = torch.randn((B,T,C))


# print(loss.item())
# print(decode(m.generate(idx=torch.zeros((1,1), dtype = torch.long),max_new_tokens=100)[0].tolist()))

xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev, 0)


# wei = torch.tril(torch.ones(T,T))
# # print(wei)
# wei = wei/ torch.sum(wei,1,keepdims = True)
# xbow2 = wei @ x
# print(torch.allclose(xbow,xbow2))

# tri = torch.tril(torch.ones(T,T))
# wei = torch.zeros((T,T))
# # print(wei)
# wei = wei.masked_fill(tri==0, float('-inf'))
# wei = F.softmax(wei, dim=-1)
# xbow3 = wei @ x
# print(torch.allclose(xbow,xbow3))

head_size = 16
key = nn.Linear(C, head_size, bias=True)
query = nn.Linear(C, head_size, bias=True)
value = nn.Linear(C, head_size, bias=True)
k = key(x)
q = query(x)
v = value(x)
wei = q @ k.transpose(-2,-1)
tri = torch.tril(torch.ones(T,T))
# wei = torch.zeros((T,T))
# print(wei)
wei = wei.masked_fill(tri==0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ v