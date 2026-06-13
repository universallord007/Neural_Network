import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


words = open("names.txt", "r").read().splitlines()

N = torch.zeros((27,27),dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 3
X ,Y = [],[]
for w in words[:5]:
    # print(w)
    context = [0]*block_size
    for ch in w + '.':
        ix =stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '------>' , itos[ix])
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
c = torch.randn((27,2))
# We can index the c tensor with list-indexing as well as with the tensor indexing
# print(c[torch.tensor([5,6,7])])
# print((F.one_hot(torch.tensor(5),num_classes = 27).float())@c)

emb = c[X]
w1 = torch.randn((6,100))
b1 = torch.randn(100)
# con = torch.cat(torch.unbind(emb,1),1)
# print(con)
# This is the hidden layer of activation 
h = torch.tanh((emb.view(-1,6)@w1) + b1)

w2 = torch.randn((100,27))
b2 = torch.randn(27)

logits = h @ w2 + b2
counts = logits.exp()
# prob = counts/counts.sum(1,keepdims= True)
# # print(prob[0].sum().it5em())
# # print(prob[torch.arange(32),Y])
# loss = -prob[torch.arange(32),Y].log().mean()
# This is using the cross Entropy function
parameters = [c, w1, b1, w2, b2]
loss = F.cross_entropy(logits,Y)
for p in parameters:
    p.grad = None
loss.backward()

for p in parameters:
    p += -0.1*p.grad