
import dill
import numpy as np

from matplotlib import pylab as pl
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from cycler import cycler
from itertools import chain

from dpFRAG import dpFRAG

#export_path = '/home/watkinspv/Desktop'
export_path = ''
figno = 1000
name = '23f'
get_metrics = True

fns = ['../out_classifier_lda_35iter_20f_test_6fold%d.dill' % (x,) for x in range(6)]
#fns = ['../out_classifier_lda_35iter_20f_test_all.dill']
nfns = len(fns)

for fn,j in zip(fns,range(nfns)):
    with open(fn, 'rb') as f: d = dill.load(f)
    clfs = d['classifiers']
    metrics = d['train_metrics']

    if j==0: 
        nclfs = sum([x is not None for x in clfs]); nfeatures = clfs[0].scalings_.shape[0]
        dists = np.zeros((nclfs,2,nfns),dtype=np.double)
        scalings = np.zeros((nfeatures,nclfs,nfns),dtype=np.double)
        pos = np.zeros((nclfs,nfns),dtype=np.int64)
        tpos = np.zeros((nclfs,nfns),dtype=np.int64)
        fpos = np.zeros((nclfs,nfns),dtype=np.int64)
        #precision = np.zeros((nclfs,nfns),dtype=np.double)
        #negatives = np.zeros((nclfs,nfns),dtype=np.int64)
    else: 
        assert(nclfs == len(clfs)); assert(nfeatures == clfs[0].scalings_.shape[0])
    
    for clf,i,metric in zip(clfs,range(nclfs),metrics):
        dists[i,:,j] = clf.decision_function(clf.means_)   # distance from means to the decision boundary
        scalings[:,i,j] = clf.scalings_[:,0]
        if get_metrics: 
            pos[i,j] = metric['nyes']; tpos[i,j] = metric['tp']; fpos[i,j] = metric['fp']; 

adists = np.abs(dists)
udists = np.mean(adists,axis=2)
#sdists = np.std(adists,axis=2)/np.sqrt(nfns)
sdists = np.std(adists,axis=2)

ascalings = np.abs(scalings)
uscalings = np.mean(ascalings,axis=2)
#sscalings = np.std(ascalings,axis=2)/np.sqrt(nfns)
sscalings = np.std(ascalings,axis=2)

ivals = np.arange(1,nclfs+1)

utpos = np.mean(tpos,axis=1)
stpos = np.std(tpos,axis=1)
upos = np.mean(pos,axis=1)
spos = np.std(pos,axis=1)
ufpos = np.mean(fpos,axis=1)
sfpos = np.std(fpos,axis=1)

pl.figure(figno)
axes = pl.subplot(2,2,1)
pl.plot(ivals,udists[:,0],'r',linewidth=2); pl.plot(ivals,udists[:,1],'g',linewidth=2);
pl.plot(ivals,udists[:,0]+sdists[:,0],'r--'); pl.plot(ivals,udists[:,0]-sdists[:,0],'r--')
pl.plot(ivals,udists[:,1]+sdists[:,1],'g--'); pl.plot(ivals,udists[:,1]-sdists[:,1],'g--');
plt.xlim([ivals[0]-1,ivals[-1]+1])
pl.xlabel('iteration'); pl.ylabel('distance between class mean and hyperplane')
pl.title('yes (g), not (r)')
            
#subfeatures = [list(chain(range(9), range(18,21)))+[22], list(range(9,18))+[21]]; bbox = [(1.1, -0.2), (1.5, 0.9)]
#subfeatures = [list(chain(range(9), range(18,23))), list(range(9,18))]; bbox = [(1.1, -0.2), (1.5, 0.9)]
#subfeatures = [list(chain(range(6), range(10,15))), list(chain(range(6,9), range(15,20)))]; bbox = [(1.1, -0.2), (1.5, 0.9)]
subfeatures = [[0,1,2,3,4,10,12,13,14,16], [5,6,7,8,9,11,15,17,18,19]]; bbox = [(1.1, -0.2), (1.5, 0.9)]
#subfeatures = [range(9), range(9,14)]; bbox = [(1.1, -0.2), (1.5, 0.9)]
#subfeatures = [range(nfeatures)]; bbox = [(1.1, -0.2)]
x = uscalings.T; sl = uscalings.T-sscalings.T; sh = uscalings.T+sscalings.T
for i,fsel in zip(range(len(subfeatures)),subfeatures):
    axes = pl.subplot(2,2,2+i)
    cnfeatures = len(fsel)
    scalarMap = mpl.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=cnfeatures), cmap=plt.get_cmap('Set1'))
    colorVal = [scalarMap.to_rgba(x) for x in range(cnfeatures)]
    axes.set_prop_cycle(cycler('color', colorVal))
    plt.ylabel('eigen scaling'); plt.xlabel('iteration');
    pl.plot(ivals,x[:,fsel],linewidth=2);
    pl.plot(ivals,sl[:,fsel],linestyle='--',linewidth=1)
    pl.plot(ivals,sh[:,fsel],linestyle='--',linewidth=1)
    plt.xlim([ivals[0]-1,ivals[-1]+1])
    plt.legend([dpFRAG.FEATURES_NAMES[x] for x in fsel],bbox_to_anchor=bbox[i])

if nfns==1 and len(subfeatures)==1:
    axes = pl.subplot(2,2,3)
    scalarMap = mpl.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=nclfs), cmap=plt.get_cmap('jet'))
    colorVal = [scalarMap.to_rgba(x) for x in range(nclfs)]
    axes.set_prop_cycle(cycler('color', colorVal))
    #axes.set_prop_cycle(cycler('lw', (np.arange(nfeatures,dtype=np.double)/3+1/3)[::-1]))
    plt.xticks(range(nfeatures),rotation=45); axes.set_xticklabels(dpFRAG.FEATURES_NAMES)
    plt.ylabel('eigen scaling');
    pl.plot(np.squeeze(scalings))
    plt.xlim([-1,nfeatures])

pl.figure(figno+1)
axes = pl.subplot(1,2,1)
pl.plot(ivals,np.log10(utpos),'b')
pl.plot(ivals,np.log10(utpos-stpos),'b--'); pl.plot(ivals,np.log10(utpos+stpos),'b--')
pl.plot(ivals,np.log10(upos),'g')
pl.plot(ivals,np.log10(upos-spos),'g--'); pl.plot(ivals,np.log10(upos+spos),'g--')
plt.ylabel('log positives'); plt.xlabel('iteration'); plt.title('total (b) true (g)')
plt.xlim([ivals[0]-1,ivals[-1]+1])
axes = pl.subplot(1,2,2)
pl.plot(ivals,ufpos,'r')
pl.plot(ivals,ufpos-sfpos,'r--'); pl.plot(ivals,ufpos+sfpos,'r--')
plt.ylabel('false positives'); plt.xlabel('iteration');
plt.xlim([ivals[0]-1,ivals[-1]+1])

if export_path:
    figna = [x % (name,) for x in ['iterative_lda_%s.png']]
    nfigna = len(figna)
    for f,i in zip(range(figno, figno+nfigna), range(nfigna)):
        pl.figure(f)
        figure = plt.gcf() # get current figure
        figure.set_size_inches(20, 20)
        plt.savefig(os.path.join(export_path,figna[i]), dpi=72)
else:
    pl.show()

