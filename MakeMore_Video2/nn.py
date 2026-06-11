import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


words = open("names.txt", "r").read().splitlines()

N = torch.zeros((27,27),dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0


xs , ys = [] , []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
xenc = F.one_hot(xs , num_classes = 27).float()
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator = g , requires_grad = True)
# Forward Pass
for i in range(1000):
    logits = xenc @ W
    counts = logits.exp()
    prob = counts / counts.sum(1, keepdim = True)
    loss = -prob[torch.arange(len(ys)), ys].log().mean()

    # Backward Pass
    W.grad = None
    loss.backward()
    W.data += -50*W.grad

print(loss.item())
