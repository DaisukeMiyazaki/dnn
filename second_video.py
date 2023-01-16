import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

# read words 
words = open("names.txt", "r").read().splitlines()
words[:8]

# build the vocab of chars and mapping to/from int
chars = sorted(list(set(''.join(words))))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi['.'] = 0
itos = {i: ch for ch, i in stoi.items()}
print(itos)


def build_dataset(words):

    block_size = 3 
    X_input, Y_label = [], []
    for word in words:
        context = [0] * block_size
        for ch in word + '.':
            ix = stoi[ch]
            X_input.append(context)
            Y_label.append(ix)
            # print(''.join(itos[i] for i in context), '---->', itos[ix])
            # 以下debug用
            # print(context, '---->', ix)
            # print(context[1:], '---->', [ix])
            context = context[1:] + [ix] # crop and append

    X_input = torch.tensor(X_input, dtype=torch.int64)
    Y_label = torch.tensor(Y_label, dtype=torch.int64)
    print(X_input.shape, Y_label.shape)
    return X_input, Y_label

import random
random.seed(42)
random.shuffle(words)
n1 = int(len(words) * 0.8)
n2 = int(len(words) * 0.9)

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

## 以下、モデル化   
print(Xtr.shape, Ytr.shape)

g = torch.Generator().manual_seed(214483647)
C = torch.randn((27, 18), generator=g)
W1 = torch.randn((18, 180), generator=g)
b1 = torch.randn((180,), generator=g)
W2 = torch.randn((180, 27), generator=g)
b2 = torch.randn((27,), generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

lrp = torch.linspace(-3, 0, 1000)
lre = 10**lrp

def forward(parameters):
    return sum(p.nelement() for p in parameters) # the number of parameters

lri = []
lossi = []
stepi = []


for i in range(50000):

    # minibatch construction
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 18) @ W1 + b1)
    logits = h @ W2 + b2
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdim=True)
    # loss = - prob[torch.arange(32), Y_label].log().mean() -> これは、以下のように書き換えられる。forward pass, backward pass, fewer kernel calls, and less memory usage
    loss = F.cross_entropy(logits, Ytr[ix])
    # print(loss)
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # update 
    # lr = lre[i] 下の方法でlrの幅の予測ができる
    lr = 0.05 #　最後のほうに使う learning rate, 最後のほうはdecayする
    # lr = 0.1 if i < 30000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # tracking
    # lri.append(lrp[i])
    lossi.append(loss.log10().item())
    stepi.append(i)

print(loss.item())

plt.plot(stepi, lossi)

def test_accuracy(X, Y):
    emb = C[X]
    h = torch.tanh(emb.view(-1, 18) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    return loss
