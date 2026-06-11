import torch
import matplotlib.pyplot as plt
words = open("names.txt","r").read().splitlines()
# print(min(len(w) for w in words))
# print(max(len(w) for w in words))


b = {}
# print(sorted(b.items(), key = lambda kv : -kv[1]))

N = torch.zeros((27,27),dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] +=1



itos = {i:s for s,i in stoi.items()}

plt.figure(figsize=(24, 24))  # bigger figure
plt.imshow(N, cmap='Blues')

for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray', fontsize=7)
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray', fontsize=7)

# plt.axis('off')
# plt.tight_layout()
# plt.savefig('bigram.png', dpi=200, bbox_inches='tight')  # save high-res
# plt.show()

p = N[0].float()
# p = p/p.sum()
g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()
torch.multinomial(p, num_samples=20, replacement=True, generator=g)


g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
P = (N+1).float()
P /=  P.sum(1,keepdim=True)
# print(P[0].sum())
g = torch.Generator().manual_seed(2147483647)
for _ in range(10):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    # print(''.join(out)
    # )

# This will act as our training loop to enhance the model's response
log_like = 0.0
n = 0
for w in ["conorzz"]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] +=1
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
        n += 1
        log_like += logprob
        print(f'{ch1}{ch2} with probability of {prob :.4f} and in log it is {logprob : .4f}')

print(log_like.item())
nll = -log_like
print(f'{nll/n}')