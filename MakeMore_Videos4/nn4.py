import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# Build dataset function
block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr,  Ytr  = build_dataset(words[:n1])   # 80% training
Xdev, Ydev = build_dataset(words[n1:n2]) # 10% dev/validation
Xte,  Yte  = build_dataset(words[n2:])  # 10% test

g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((27, 10),   generator=g)
W1 = torch.randn((30, 200),  generator=g)*((5/3)/(30**0.5))
b1 = torch.randn(200,       generator=g)*0.01
W2 = torch.randn((200, 27), generator=g)*((5/3)/(200**0.5))
b2 = torch.randn(27,        generator=g)*0

bngain = torch.ones((1,200))
bnbias = torch.zeros((1,200))
bnmean_running = torch.zeros((1,200))
bnstd_running = torch.ones((1,200))
parameters = [C, W1, b1, W2, b2, bngain, bnbias]


# print(sum(p.nelement() for p in parameters)) # total parameters

for p in parameters:
    p.requires_grad = True

# This is the training split
stepi = []
lossi = []
for i in range(50000):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    emb = C[Xtr[ix]]
    embcat = emb.view(-1, 30)
    hpreact = embcat @ W1 + b1
    bnmeani = hpreact - hpreact.mean(0,keepdims=True)
    bnstdi =  hpreact.std(0,keepdims=True)
    hpreact = bngain*(bnmeani)/bnstdi + bnbias
    with torch.no_grad():
        bnmean_running = 0.99*bnmean_running + 0.01*bnmeani
        bnstd_running = 0.99*bnstd_running + 0.01*bnstdi
    h = torch.tanh(hpreact )
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < 10000 else 0.01
    stepi.append(i)
    lossi.append(loss.item())
    for p in parameters:
        p.data += -lr * p.grad
    if i % 10000 == 0:
        lossi.append(loss.log10().item())
        # print("Loss is", loss.log10().item())
print(loss.item())
# plt.plot(stepi,lossi)
# plt.show()

# This is the dev split
with torch.no_grad():
    emb = C[Xtr]
    emb2 = emb.view(-1, 30)
    hpreact1 = emb2 @ W1 + b1
    hpreact1 = bngain*(hpreact - hpreact.mean(0,keepdims=True))/hpreact.std(0,keepdims=True) + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr)
    print(loss.item())

# emb = C[Xdev]
# h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
# logits = h @ W2 + b2
# loss = F.cross_entropy(logits, Ydev)
# print(loss.item())

# plt.figure(figsize=(8,8))
# plt.scatter(C[:,0].data, C[:,1].data, s=200)
# for i in range(C.shape[0]):
#     plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
# plt.grid('minor')
# plt.show()

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size  # initialize with all ...
    while True:
        emb = C[torch.tensor([context])]  # (1,block_size,d)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))

# plt.plot(torch.tanh())
# plt.show()
# x = torch.randn(1000,20)
# w = torch.randn(10,200)/ 10**0.5
# y = w@x
# print(x.mean(), x.std())
# print(y.mean(), y.std())
# Exercise 1: backprop through the whole thing manually, 
# backpropagating through exactly all of the variables 
# as they are defined in the forward pass above, one by one

# -----------------
# YOUR CODE HERE :)
# -----------------
# dlogprobs = torch.zeros_like(logprobs)
# dlogprobs[range(n), Yb] = -1.0/n
# dprobs = 1/probs * dlogprobs
# dcounts_sum_inv = (counts * dprobs).sum(1, keepdims = True)
# dcounts = counts_sum_inv * dprobs
# dcounts_sum = -counts_sum**-2 * dcounts_sum_inv
# dcounts += torch.ones_like(counts)* dcounts_sum
# dnorm_logits = counts * dcounts
# dlogits = dnorm_logits.clone()
# dlogit_maxes  = - dnorm_logits.sum(1,keepdims = True)
# dlogits += F.one_hot(logits.max(1).indices, num_classes = logits.shape[1])*dlogit_maxes
# dh = dlogits @ W2.T
# dW2 = h.T @ dlogits
# db2 = dlogits.sum(0)
# dhpreact = (1.0 - h**2)*dh
# dbngain = (bnraw * dhpreact).sum(0,keepdims = True)
# dbnraw = dhpreact * bngain
# dbnbias = dhpreact.sum(0)
# bnmeani = 1/n*hprebn.sum(0, keepdim=True)
# dbnvar_inv = (bndiff * dbnraw).sum(0,keepdims = True)
# dbndiff = bnvar_inv * dbnraw
# dbnvar = (-0.5 * (bnvar + 1e-5)**-1.5)* dbnvar_inv
# dbndiff2 = (1.0/(n-1)* torch.ones_like(bndiff2))* dbnvar
# dbndiff += 2*bndiff * dbndiff2
# dhprebn = dbndiff.clone()
# dbnmeani= (-dbndiff).sum(0)
# dhprebn += 1.0/n * torch.ones_like(hprebn) * dbnmeani
# dembcat = dhprebn @ W1.T
# dW1 = embcat.T @ dhprebn
# db1 = dhprebn.sum(0)
# demb = dembcat.view(emb.shape)
# dC = torch.zeros_like(C)
# for i in range(Xb.shape[0]):
#   for j in range(Xb.shape[1]):
#     ix = Xb[i,j]
#     dC[ix] += demb[i,j]
# cmp('logprobs', dlogprobs, logprobs)
# cmp('probs', dprobs, probs)
# cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
# cmp('counts_sum', dcounts_sum, counts_sum)
# cmp('counts', dcounts, counts)
# cmp('norm_logits', dnorm_logits, norm_logits)
# cmp('logit_maxes', dlogit_maxes, logit_maxes)
# cmp('logits', dlogits, logits)
# cmp('h', dh, h)
# cmp('W2', dW2, W2)
# cmp('b2', db2, b2)
# cmp('hpreact', dhpreact, hpreact)
# cmp('bngain', dbngain, bngain)
# cmp('bnbias', dbnbias, bnbias)
# cmp('bnraw', dbnraw, bnraw)
# cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
# cmp('bnvar', dbnvar, bnvar)
# cmp('bndiff2', dbndiff2, bndiff2)
# cmp('bndiff', dbndiff, bndiff)
# cmp('bnmeani', dbnmeani, bnmeani)
# cmp('hprebn', dhprebn, hprebn)
# cmp('embcat', dembcat, embcat)
# cmp('W1', dW1, W1)
# cmp('b1', db1, b1)
# cmp('emb', demb, emb)
# cmp('C', dC, C)


# backward pass

# -----------------
# YOUR CODE HERE :)
# dlogits = F.softmax(logits,1)
# dlogits[range(n),Yb]-= 1
# dlogits/=n
# # -----------------

# cmp('logits', dlogits, logits) # I can only get approximate to be true, my maxdiff is 6e-9