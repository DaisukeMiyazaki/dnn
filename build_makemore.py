# https://www.youtube.com/watch?v=PaCmpygFfXo&t=2520s
import torch
import matplotlib.pyplot as plt
%matplotlib inline


# データの読み込み
words = open("names.txt", "r").read().splitlines()

# データの確認
_min = min(len(w) for w in words)
_max = max(len(w) for w in words)
print(_min)
print(_max)

def preprocess(words):
    """
    データの前処理
    <S>, <E>はNLPでよく使われる特殊文字
    """
    frequencies = {}
    for w in words:
        special_chs = ['<S>'] + list(w) + ['<E>']
        for ch1, ch2 in zip(special_chs, special_chs[1:]):
            bigram = (ch1, ch2)
            frequencies[bigram] = frequencies.get(bigram, 0) + 1
            print(ch1, ch2)

    sorted(frequencies.items(), key=lambda key_value: key_value[1], reverse=True)
    return frequencies

chars_metrics = torch.zeros((28, 28), dtype=torch.int32)


def get_all_characters(words):
    """
    すべての文字を取得する
    """
    all_chars = sorted(list(set(''.join(words))))
    return all_chars

def get_lookup_table_chars_to_int(words):
    """
    look up table
    文字列をベクトルに変換する必要あり。整数値を持ってきたい
    文字 -> 数値の辞書を作成する
    """
    all_chars = get_all_characters(words)
    chars_to_int = {ch: i for i, ch in enumerate(all_chars)}
    chars_to_int['<S>'] = 26
    chars_to_int['<E>'] = 27
    # chars_to_int['.'] = 26
    return chars_to_int

def get_lookup_table_int_to_chars(chars_to_int):
    int_to_chars = {i: ch for ch, i in chars_to_int.items()}
    return int_to_chars

for w in words:
    special_chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(special_chs, special_chs[1:]):
        ch1_idx = chars_to_int[ch1]
        ch2_idx = chars_to_int[ch2]
        chars_metrics[ch1_idx, ch2_idx] += 1

int_to_chars = {i: ch for ch, i in chars_to_int.items()}

plt.figure(figsize=(16, 16))
plt.imshow(chars_metrics, cmap='Blues')
for i in range(28):
    for j in range(28):
        bigram_repr = int_to_chars[i] + int_to_chars[j]
        plt.text(j, i, bigram_repr, ha='center', va='bottom', color='gray')
        plt.text(j, i, chars_metrics[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
#　このグラフは、文字列の出現頻度を表している
#　例えば、'a'が'aの後に出現した回数は556回

# <S>, <E>はNLPでよく使われる特殊文字
# 代わりに . を使う
# 例: <S>hello<E> -> .hello.
# .の辞書keyは0番目には配置する
master = torch.zeros((27, 27), dtype=torch.int32)
str_to_int = {ch: i+1 for i, ch in enumerate(all_chars)}
str_to_int['.'] = 0
int_to_str = {i: ch for ch, i in str_to_int.items()}
int_to_str

for w in words:
    special_chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(special_chs, special_chs[1:]):
        ch1_idx = str_to_int[ch1]
        ch2_idx = str_to_int[ch2]
        master[ch1_idx, ch2_idx] += 1


plt.figure(figsize=(16, 16))
plt.imshow(master, cmap='Blues')
for i in range(27):
    for j in range(27):
        bigram_repr = int_to_str[i] + int_to_str[j]
        plt.text(j, i, bigram_repr, ha='center', va='bottom', color='gray')
        plt.text(j, i, master[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')


p = (master + 1).float()
p = p / p.sum(1, keepdim=True)

plt.figure(figsize=(16, 16))
plt.imshow(master, cmap='Blues')
for i in range(27):
    for j in range(27):
        bigram_repr = int_to_str[i] + int_to_str[j]
        plt.text(j, i, bigram_repr, ha='center', va='bottom', color='gray')
        plt.text(j, i, master[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
#　このグラフは、文字列の出現頻度を表している
#　例えば、'a'が'aの後に出現した回数は556回


# 確率分布ベクトルに変換する。まずはfloatにタイプを変換する
p = master[0].float()
p = p / p.sum()

g = torch.Generator().manual_seed(2147483647)
idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g) # tensor([13])
idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
int_to_str[idx]

# 確率を固定するためにgeneratorを使う
g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
print(p)
p = p / p.sum()


# replacement = True で重複を許す。Falseで重複を許さない
# pは確率分布ベクトル
# 1回のサンプリングで1つの文字を選択する
# generatorは乱数のシードを固定する
torch.multinomial(p, num_samples=100, replacement=True, generator=g)

P = (master+1).float()
P /= P.sum(dim=1, keepdim=True)
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    idx = 0
    out = []
    while True:
        p = P[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(int_to_str[idx])
        if idx == 0:
            break

    print(''.join(out))

# 評価関数は、すべての確率を掛け合わせたもの
# 例: p1 * p2 * p3 * p4
# 実際はpは小さくなるので、対数を取り、足し合わせる
# 例: log(p1) + log(p2) + log(p3) + log(p4) = log(p1 * p2 * p3 * p4)
# この関数をlog-likelihoodと呼ぶ
# 関数の合計値はマイナスなので、符号を反転してかつ標準化する

# あまりにもサンプルにないデータをいれると、確率が0になるため、すべての文字出現確率に1を足す
# これはスムージングと呼ばれる(smoothing)

log_likelihood = 0.0
n = 0

# for w in words:
for w in ['daisukeqgadlw']:
    special_chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(special_chs, special_chs[1:]):
        ch1_idx = str_to_int[ch1]
        ch2_idx = str_to_int[ch2]
        prob = P[ch1_idx, ch2_idx]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f"{ch1}, {ch2}: {prob:.3f} {logprob:.3f}")

print(f"Log likelihood: {log_likelihood}")
nll = -log_likelihood
print(f"Negative log likelihood: {nll}")
print(f"Perplexity: {nll / n}")

sorted(b.items(), key = lambda kv: -kv[1])

N = torch.zeros((28,28), dtype=torch.int32)

# create the traning set for all the bigram model (x, y)
xs, ys = [], []

for w in words[:1]:
    special_chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(special_chs, special_chs[1:]):
        ch1_idx = chars_to_int[ch1]
        ch2_idx = chars_to_int[ch2]
        print(f"{ch1}, {ch2}: {ch1_idx}, {ch2_idx}")
        xs.append(ch1_idx)
        ys.append(ch2_idx)
    
xs = torch.tensor(xs, dtype=torch.int64)
ys = torch.tensor(ys, dtype=torch.int64)
# torch.nn.functional.one_hot(my_tensor.long())　でone-hotベクトルに変換できる、torch.int32では  one_hot is only applicable to index tensor.エラー
        

import torch.nn.functional as F
x_enc = F.one_hot(xs, num_classes=27).float() # onehotはlong型を返すので、floatに変換する必要あり

plt.imshow(x_enc) # tensorのshapeが(3, 27)なので、3行27列の画像になる, tensorを可視化する。黄色が1、青が0

# 5, 27
# 27, 1
W = torch.randn(27, 1)
x_enc @ W # pytorchで@は行列の積のオペレーションをする

# 5, 27
# 27, 27 -> 27, 27
W = torch.randn(27, 27)
x_enc @ W # 27, 27

logits = (x_enc @ W) # expは指数関数, すべての要素に対してexpを計算してfloatをカウントとして読み取る, log count -> count
counts = logits.exp() # すべての要素に対してexpを計算してfloatをカウントとして読み取る, log count -> count
probs = counts / counts.sum(dim=1, keepdim=True) # すべての要素を合計して1になるように正規化する

# summary   ----------------    

# randomly generate 
g = torch.Generator().manual_seed(2147483647 + 5)
W = torch.randn((27, 27), generator=g, )

xenc = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(dim=1, keepdim=True)
# the last 2 steps can be done by softmax, which takes positive and negative numbers and normalizes them to be between 0 and 1


nlis = torch.zeros(5)
for i in range(5):
    # i-th bigram
    x = xs[i].item() # input character index
    y = ys[i].item() # input character index
    print("------------------")
    print(f"bigram example{i+1}: {int_to_chars[x]}, {int_to_chars[y]} indexes {x} and {y}")
    print("input to the nural net", x)
    print("output probability of the nural net", probs[i])
    print("label, the actual character", y)
    p = probs[i, y]
    print("probability assigned by the net to the correct character", p.item())
    logp = torch.log(p)
    print("log likelihood of the correct character", logp.item())
    nli = -logp
    print("negative log likelihood of the correct character", nli.item())
    nlis[i] = nli

print("====================================")
print("average negative log likelihood, loss =", nlis.mean().item())

#forward pass
# randomly generate 
g = torch.Generator().manual_seed(2147483647 + 5)
W = torch.randn((27, 27), generator=g, requires_grad=True) # requires_grad=Trueで勾配を計算する, defaultでfalse


#optimization part

xenc = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(dim=1, keepdim=True)
# the last 2 steps can be done by softmax, which takes positive and negative numbers and normalizes them to be between 0 and 1
loss = -probs[torch.arange(5), ys].log().mean()

#backpropagation
W.grad = None #set to zero the gradient
loss.backward()



# create data set
xs = []
ys = []

nlis = torch.zeros(5)
for w in words:
    # i-th bigram
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(chars_to_int[ch1])
        ys.append(chars_to_int[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print("number of examples", num)

g = torch.Generator().manual_seed(2147483647 + 5)
W = torch.randn((27, 27), generator=g, requires_grad=True) # requires_grad=Trueで勾配を計算する, defaultでfalse

(W ** 2).sum() # regularization

for k in range(100):
    
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    # the last 2 steps can be done by softmax, which takes positive and negative numbers and normalizes them to be between 0 and 1
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()
    print(loss.item())

    #backpropagation
    W.grad = None #set to zero the gradient
    loss.backward()

    W.data += -50 * W.grad # update the weights


# how to sample from the model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):

    out = []
    ix = 0
    while True:

        # before
        # p = P[ix]
        # forward pass
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)

        # sample from the model
        ix = torch.multinomial(p, num_samples=1, generator=g).item()
        out.append(int_to_chars[ix])
        if ix == 0:
            break

    print("".join(out))
