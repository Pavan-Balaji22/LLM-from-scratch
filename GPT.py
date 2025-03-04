
import torch
import torch.nn as nn
from torch.nn import functional as F
import mlflow as mlflow

# Hyper Parameters
batch_size = 64
block_size = 256
lrsloss = []
lossi = []
lr = 1e-3
max_iter = 5000
step_iter = 500
eval_iter = 200
n_embed = 512
n_heads = 32
n_layer = 8
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

shakspheredata= open("input.txt",mode="r",encoding="utf8").read()
vocab = sorted(list(set(shakspheredata)))
nvocab = len(vocab)

stoi = {k:v for v,k in enumerate(vocab)}
itos = {v:k for v,k in enumerate(vocab)}
encode = lambda x:[stoi[i] for i in x]
decode = lambda x: "".join([itos[i] for i in x])

#Data preparation
text = torch.tensor(encode(shakspheredata))
n = int(.9*len(text))
train = text[:n]
val = text[n:]



@torch.no_grad()
def esitimate_loss():
    out = {}
    model.eval()
    for mode in ["train","val"]:
        losses = 0
        for i in range(eval_iter):
            X,Y = get_batch(mode)
            _,lossb = model(X,Y)
            losses = losses+lossb
        out[mode] = losses/eval_iter
    model.train()
    return out

def get_batch(split:str):
    data = train if split == "train" else val
    ix =  torch.randint(len(data) - block_size ,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size ,bias = False)
        self.query = nn.Linear(n_embed,head_size ,bias = False)
        self.value = nn.Linear(n_embed,head_size ,bias = False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        B,T,C = x.shape
        kx = self.key(x) #B,T, headsize
        qx = self.query(x) #B,T, headsize
        self.wei = qx @ kx.transpose(-2,-1) *kx.shape[-1] ** -0.5 # transpose last 2 dimension
        # self.wei = self.wei.tril()
        self.wei= self.wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        self.wei = F.softmax(self.wei,dim=-1)
        self.wei = self.dropout(self.wei)
        v = self.value(x)
        out = self.wei @  v 

        return out

class Multihead(nn.Module):
    def __init__(self):
        super().__init__()
        headsize = n_embed//n_heads
        self.Multihead = nn.ModuleList([Head(headsize) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        out = torch.cat([h(x) for h in self.Multihead], dim=-1)
        out = self.dropout(self.proj(out))
        return out
class FForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linearModel =  nn.Sequential(nn.Linear(n_embed,4 *n_embed),
                                          nn.ReLU(),
                                          nn.Linear(4 * n_embed,n_embed),
                                          nn.ReLU(),
                                          nn.Dropout(dropout))

    def forward(self,x):
        return self.linearModel(x)
class TransformerDecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Multihead()
        self.ffnn = FForward()
        self.layern1 = nn.LayerNorm(n_embed)
        self.layern2 = nn.LayerNorm(n_embed)
    def forward(self,x):
        x = x + self.attention(self.layern1(x))
        out = x + self.ffnn(self.layern2(x))
        return out
    
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # print(nvocab,n_embed,block_size)
        self.embedding_table = nn.Embedding(nvocab,n_embed)
        self.positional_embedding = nn.Embedding(block_size,n_embed)
        self.transformerdecode = nn.Sequential(*[TransformerDecoderBlock() for x in range(n_layer)])
        self.layern1 = nn.LayerNorm(n_embed)
        self.linear1 = nn.Linear(n_embed,nvocab)
    
    def forward(self,idx,target=None):
        
        B,T, = idx.shape
        tok_embed = self.embedding_table(idx) 
        pos_embed = self.positional_embedding(torch.arange(T,device=device))
        x = tok_embed + pos_embed
        x = self.transformerdecode(x)
        x = self.layern1(x)
        logits = self.linear1(x)

        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = target.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss

    def generate(self,max_tokens:int,idx:torch.Tensor):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits,loss = self(idx_cond) # B,T,C
            logits = logits[:,-1,:] #B,C Picking last time step
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,1)
            idx =torch.cat((idx,idx_next), dim=1)

        return idx


model = GPT()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
#Training
print("Start Training")
for i in range(max_iter):
    if i % step_iter== 0:
        eloss = esitimate_loss()
        print(f"Loss at {i}: Train loss: {eloss['train']} | Validation loss :{eloss['val']}")

    #Forward Pass
    xb,yb = get_batch('train')
    
    logits,loss = model(xb,yb)
    
    #Backpass
    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()

print(F" Final loss: {loss.item():.4f}")

#Generation
print(decode(model.generate(1000,idx = torch.zeros((1,1),dtype=torch.long))[0].tolist()))


