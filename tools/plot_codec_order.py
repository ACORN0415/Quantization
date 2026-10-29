"""Static export: each cell shows p0/p4 D ratios, color shows worse prompt."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'src/experiments/codec_order_development'
r=list(csv.DictReader((OUT/'summary.csv').open()))
methods=('tarp','tsvd','pivot_qr','range','tucker','tt')
orders=('method_only','quant_then_method','method_then_quant_rank','method_then_quant_budget')
labels=('Method only\nFP16','Quantize ->\nmethod','Method -> quantize\nsame rank','Method -> quantize\nsame budget')
fig,axs=plt.subplots(1,3,figsize=(17,5.5),layout='constrained')
for ax,b in zip(axs,(2,3,4)):
 grid=np.zeros((6,4));vals={}
 for i,m in enumerate(methods):
  for j,o in enumerate(orders):
   v=[float(next(x['ratio'] for x in r if int(x['bits'])==b and x['method']==m and x['order']==o and int(x['prompt'])==p)) for p in (0,4)]
   grid[i,j]=max(v);vals[i,j]=v
 im=ax.imshow(grid,norm=LogNorm(.5,100),cmap='RdYlGn_r',aspect='auto')
 ax.set_title(f'{b}-bit RTN storage budget')
 ax.set_xticks(range(4),labels,fontsize=8);ax.set_yticks(range(6),methods)
 for (i,j),v in vals.items():ax.text(j,i,f'{v[0]:.2f} / {v[1]:.2f}',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=axs,label='Worse prompt ratio (log scale): lower is better',shrink=.8)
fig.suptitle('Final output squared-error / plain RTN error   |   cells: p0 / p4\nDevelopment V-only diagnostic; not video quality',fontsize=13)
fig.savefig(OUT/'comparison.png',dpi=180)
fig.savefig(OUT/'comparison.svg')
