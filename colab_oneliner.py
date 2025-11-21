"""
COLAB ONE-LINER: Copy everything below this line into a single Colab cell
"""

!pip install torch numpy -q

import torch, torch.nn as nn, torch.nn.functional as F, os, json, random, copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

class TransparentRNN(nn.Module):
    def __init__(s, i, h, o): super().__init__(); s.ip=nn.Linear(i,h); s.gru=nn.GRU(h,h,batch_first=True); s.bh=nn.Linear(h,o); s.ah=nn.Linear(h,o)
    def forward(s,x,h=None): p=F.relu(s.ip(x)); o,_=s.gru(p,h if h is not None else torch.zeros(1,x.shape[0],s.gru.hidden_size,device=x.device)); return {'beliefs':torch.sigmoid(s.bh(o[:,-1])),'actions':torch.sigmoid(s.ah(o[:,-1]))}

class RecursiveSelfAttention(nn.Module):
    def __init__(s,i,h,o,n=4): super().__init__(); s.ip=nn.Linear(i,h); s.attn=nn.MultiheadAttention(h,n,batch_first=True); s.bh=nn.Linear(h,o); s.ah=nn.Linear(h,o)
    def forward(s,x,h=None): p=F.relu(s.ip(x)); a,_=s.attn(p,p,p); return {'beliefs':torch.sigmoid(s.bh(a[:,-1])),'actions':torch.sigmoid(s.ah(a[:,-1]))}

class TransformerAgent(nn.Module):
    def __init__(s,i,h,o,l=2): super().__init__(); s.ip=nn.Linear(i,h); s.tf=nn.TransformerEncoder(nn.TransformerEncoderLayer(h,4,batch_first=True),l); s.bh=nn.Linear(h,o); s.ah=nn.Linear(h,o)
    def forward(s,x,h=None): t=s.tf(F.relu(s.ip(x))); return {'beliefs':torch.sigmoid(s.bh(t[:,-1])),'actions':torch.sigmoid(s.ah(t[:,-1]))}

@dataclass
class Agent:
    id:int; arch:str; model:nn.Module; fit:float=0.0; sa:float=0.0; zd:float=0.0
    def copy_from(s,o): s.fit,s.sa,s.zd=o.fit,o.sa,o.zd

class World:
    def __init__(s,n=6,z=2): s.n=n; s.reset(); s.zombies=set(random.sample(range(n),z))
    def reset(s): s.zombies=set(random.sample(range(s.n),2))
    def is_zombie(s,i): return i in s.zombies

def create_model(arch,i=191,h=128,o=181,dev='cpu'):
    m={'TRN':TransparentRNN,'RSAN':RecursiveSelfAttention,'Transformer':TransformerAgent}
    return m.get(arch,RecursiveSelfAttention)(i,h,o).to(dev)

def test_sally_anne(agent,dev='cpu'):
    s=torch.zeros(1,4,191,device=dev); s[0,0,0]=s[0,0,1]=s[0,0,2]=1; s[0,1,1]=s[0,1,2]=1; s[0,2,1]=s[0,2,3]=1; s[0,3,0]=s[0,3,1]=s[0,3,3]=1
    b=agent.model(s+torch.randn_like(s)*0.05)['beliefs']; return min(1,max(0,0.5+(b[0,0].item()-b[0,1].item()))) if b.shape[-1]>1 else 0.5

def test_zombie(agent,world,dev='cpu',seq=10):
    c,t=0,0
    for i in range(world.n):
        z=world.is_zombie(i); obs=torch.zeros(1,seq,191,device=dev)
        for j in range(seq): obs[0,j,0]=random.random() if z else 0.5+random.gauss(0,0.1); obs[0,j,5]=0 if z else 0.8
        p=agent.model(obs)['beliefs'][0,5].item()<0.5 if agent.model(obs)['beliefs'].shape[-1]>5 else False
        c+=int(p==z); t+=1
    return c/max(t,1)

def evolve(pop,elite=2,mut=0.15,dev='cpu'):
    pop.sort(key=lambda x:-x.fit); new=[Agent(a.id,a.arch,copy.deepcopy(a.model)) for a in pop[:elite]]
    for a in new[:elite]: a.copy_from(pop[pop.index(next(p for p in pop if p.id==a.id))])
    while len(new)<len(pop):
        p=random.choice(pop[:len(pop)//2]); c=Agent(len(new)+100,p.arch,copy.deepcopy(p.model))
        if random.random()<mut:
            with torch.no_grad():
                for param in c.model.parameters(): param.add_(torch.randn_like(param)*0.02)
        new.append(c)
    return new

def train(gens=10,pop_size=12,dev='cpu'):
    print(f"ToM-NAS Training: {gens} generations, {pop_size} agents, device={dev}")
    archs=['TRN','RSAN','Transformer']; pop=[Agent(i,archs[i%3],create_model(archs[i%3],dev=dev)) for i in range(pop_size)]
    world=World(); best_fit,best_sa,best_zd=0,0,0

    for g in range(gens):
        print(f"\n=== Generation {g} ===")
        for a in pop:
            a.model.eval()
            with torch.no_grad():
                a.sa=sum(test_sally_anne(a,dev) for _ in range(10))/10
                a.zd=sum(test_zombie(a,world,dev) for _ in range(5))/5
                a.fit=0.5*a.sa+0.5*a.zd
            print(f"  Agent {a.id} ({a.arch}): SA={a.sa:.3f} ZD={a.zd:.3f} FIT={a.fit:.3f}")

        fits=[a.fit for a in pop]; sas=[a.sa for a in pop]; zds=[a.zd for a in pop]
        print(f"Best={max(fits):.3f} Avg={sum(fits)/len(fits):.3f} SA={sum(sas)/len(sas):.3f} ZD={sum(zds)/len(zds):.3f}")
        if max(fits)>best_fit: best_fit,best_sa,best_zd=max(fits),max(pop,key=lambda x:x.fit).sa,max(pop,key=lambda x:x.fit).zd
        pop=evolve(pop,dev=dev)

    print(f"\n=== COMPLETE ===\nBest Fitness: {best_fit:.4f}\nBest Sally-Anne: {best_sa:.4f}\nBest Zombie Det: {best_zd:.4f}")
    return {'best_fitness':best_fit,'best_sally_anne':best_sa,'best_zombie':best_zd}

# RUN TRAINING
if __name__=="__main__" or True:
    results = train(gens=10, pop_size=12, dev='cuda' if torch.cuda.is_available() else 'cpu')
