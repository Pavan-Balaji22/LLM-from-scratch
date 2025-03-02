
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Sequential, Tanh, Parameter
import mlflow as mlflow

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

# Hyper Parameters
batch_size = 32
block_size = 8
lrsloss = []
lossi = []
lr = 1e-3
max_iter = 10000
step_iter = 1000
eval_iter = 200
n_embed = 32

@torch.no_grad()
def esitimate_loss(model:nn.Module):
    loss_dict = {}
    model.eval()
    for mode in ["train","val"]:
        loss = 0
        for i in range(eval_iter):
            X,Y = get_batch(mode)
            _,lossb = model(X,Y)
            loss = loss + lossb
        loss_dict[mode] = loss/eval_iter
    model.train()
    return loss_dict

def get_batch(split:str):
    data = train if "train" else val
    ix =  torch.randint(len(data) - block_size ,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x,y

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size ,bias = False)
        self.query = nn.Linear(n_embed,head_size ,bias = False)
        self.value = nn.Linear(n_embed,head_size ,bias = False)
    
    def forward(self,x):
        kx = self.key(x) #B,T, n_head
        qx = self.query(x) #B,T, n_head
        self.wei = kx @ qx.transpose(-2,-1) # transpose last 2 dimension
        self.wei = self.wei.tril()
        self.wei= self.wei.masked_fill(self.wei==0,float('-inf'))
        self.wei = torch.softmax(self.wei,dim=1)
        out = self.wei @ self.value(x)

        return out

class Multihead(nn.Module):
    def __init__(self,nheads):
        super().__init__()
        self.headsize = n_embed // nheads
        self.Multihead = nn.ModuleList([Head(self.headsize) for _ in range(nheads)])
    
    def forward(self,x):
        for head in self.Multihead:
            out = head(x)
        return out
class FForward(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.linearModel =  nn.Sequential(nn.Linear(input,output),nn.ReLU())

    def forward(self,x):
        return self.linearModel(x)
class TransformerDecoderBlock(nn.Module):
    def __init__(self,nheads):
        super().__init__()
        self.attention = Multihead(nheads)
        self.ffnn = FForward(n_embed,nvocab) 
    def forward(self,x):
        out = self.attention(x)
        out = self.ffnn(out)
        return out
    
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(nvocab,n_embed)
        self.positional_embedding = nn.Embedding(block_size,n_embed)
        self.linear1 = nn.Linear(n_embed,nvocab)

    def forward(self,idx:torch.Tensor,target:torch.Tensor=None):
        
        B,T, = idx.shape
        tok_embed = self.embedding_table(idx) 
        pos_embed = self.positional_embedding(torch.arange(T))
        x = tok_embed + pos_embed
        logits = self.linear1(x)

        if target == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            loss = F.cross_entropy(logits,target.view(B*T))
        return logits,loss

    def generate(self,max_tokens:int,idx:torch.Tensor):
        for _ in range(max_tokens):
            logits,loss = self(idx) # B,T,C
            logits = logits[:,-1,:] #B,C Picking last time step
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,1)
            idx =torch.cat((idx,idx_next), dim=1)

        return idx


model = GPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

#Training
print("Start Training")
for i in range(max_iter):
    
    #Forward Pass
    xb,yb = get_batch("train")
    logits,loss = model(xb,yb)
    lossi.append(loss.log10().item())
    
    if i % step_iter== 0:
        eloss = esitimate_loss(model)
        print(f"Loss at {i}: Train loss: {eloss['train']} | Validation loss :{eloss['val']}")
    
    #Backpass
    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()

print(F" Final loss: {loss.item():.4f}")

#Generation
print(decode(model.generate(1000,idx = torch.zeros((1,1),dtype=torch.long))[0].tolist()))


