"""Plot both fixed deadlines and the paired K8 effect on the same fresh cohort."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root=Path('results/policy_closing_confirmation')
base=json.loads((root/'comparison/results.json').read_text())
extended=json.loads((root/'horizon150/results.json').read_text())
reviews={h:json.loads((root/('review.json' if h==100 else 'horizon150/review.json')).read_text()) for h in (100,150)}
assert base['complete'] and extended['complete']
fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
forces=[0.,.2,.3,.4]
labels={'retained':'Retained','closing':'Closing candidate'}
for name,color in [('retained','#277da8'),('closing','#358d5b')]:
    for horizon,style in [(100,'--'),(150,'-')]:
        counts=[base['by_force'][f'{f:g}']['summary'][f'{name}_k8']['success_at_end_count'] if horizon==100 else
                extended['by_force'][f'{f:g}'][f'{name}_k8']['success_at_extended_horizon'] for f in forces]
        axes[0].plot(forces,np.asarray(counts)/100*100,marker='o',linestyle=style,color=color,label=f'{labels[name]}, {horizon}')
    rows=[e for e in extended['episodes'] if e['checkpoint']==name and e['chunk_length']==8]
    flags=np.zeros((len(rows),151),dtype=bool)
    for i,e in enumerate(rows):flags[i,e['success_steps']]=True
    axes[1].plot(np.arange(151)/20,flags.mean(0)*100,color=color,label=labels[name])
axes[0].set(title='K8 final success by force',xlabel='Horizontal force (N)',ylabel='Final success (%)',xticks=forces,ylim=(0,105))
axes[0].legend(fontsize=9)
axes[1].axvline(5,color='black',alpha=.5,linestyle='--')
axes[1].set(title='K8: current success over time',xlabel='Simulated task time (s)',ylabel='Official success at this step (%)',xlim=(0,7.5),ylim=(0,105))
axes[1].legend()
effects=[reviews[h]['contrasts']['closing_vs_retained']['overall']['8'] for h in (100,150)]
means=np.array([e['difference'] for e in effects])*100
cis=np.array([e['ci95'] for e in effects])*100
axes[2].errorbar([0,1],means,yerr=np.stack([means-cis[:,0],cis[:,1]-means]),fmt='o',capsize=5,color='#277da8')
axes[2].axhline(0,color='black',alpha=.5,linestyle='--')
axes[2].set(title='Paired K8 final-success effect',xticks=[0,1],xticklabels=['100 steps','150 steps\n(primary)'],ylabel='Closing − retained (percentage points)')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Closing-policy confirmation: 100 fresh matched reset groups × four forces\nFixed update-5,000 EMA; pointwise paired group-bootstrap 95% intervals')
fig.savefig(root/'comparison_horizons.png',dpi=180)
fig.savefig(root/'comparison_horizons.svg')
plt.close(fig)
