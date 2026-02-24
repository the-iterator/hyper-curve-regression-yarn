# Parametric Curve (PARCUR) model applied to Yarn dataset
# Neil Budko (c) 2026

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as hcluster

dpi = 100

# loading data
df = pd.read_csv('yarn-data.csv', delimiter=',')
dfTrain = df.loc[df['train'] == 1]
dfTest = df.loc[df['train'] == 0]

dfs = dfTrain.sort_values('density',ascending=True)
n,p = dfs.shape
p = p-3
dataset = dfs.to_numpy()
print('features, p =',p)
print('training samples, n =',n)

X = dataset[:,1:p+1]
y = np.zeros((n,1))
y[:,0] = np.copy(dataset[:,-2])

# normalizing y
ymean = np.mean(y)
y = y - ymean
ymax = np.max(np.abs(y))
y = y/ymax
S = np.concatenate((y,X),axis=1)

def getAr(S,r):
    y = S[:,0]
    V = np.vander(y,r,increasing=True)
    VtV = np.matmul(np.transpose(V),V)
    try:
        Ar = np.linalg.solve(VtV,np.matmul(np.transpose(V),S))
    except Exception as e:
        print('solving for Ar failed')
        print(f"Error: {e}")
    return Ar

def getShat(Ar,S):
    X = S[:,1:]
    Ahat = Ar[:,1:]
    AAt = np.matmul(Ahat,np.transpose(Ahat))
    GammaHat = np.linalg.solve(AAt,Ar)
    BetaHat = np.matmul(np.transpose(Ahat),GammaHat)
    Shat = np.matmul(X,BetaHat)
    return Shat

def getYclusters(S,thresh):
    n,p = S.shape
    y = np.zeros((n,1))
    y[:,0] = S[:,0]
    clusters = hcluster.fclusterdata(y, thresh, criterion="distance")
    numclst = len(np.unique(clusters))
    return clusters

def funcCV(S,fold,rmin,rmax):
    res = np.zeros((fold,rmax-rmin+1))
    for ii in range(rmax-rmin+1):
        r = rmin+ii
        rep = 0
        while rep<fold:
            Strn,Sval = splitTV(S,r)
            try:
                Ar = getAr(Strn,r)
                Shat  = getShat(Ar,Sval)
                res[rep,ii] = np.linalg.norm(Sval-Shat)/np.linalg.norm(Sval)
                rep = rep+1
            except:
                print('bad split, trying again')
    return res

def splitTV(S,r):
    n,p = S.shape
    ind_all = np.arange(n)
    ind_trn = np.random.choice(ind_all,size=int(n/3),replace=False)
    ind_val = np.delete(ind_all,ind_trn)
    Strn = S[ind_trn,:]
    Sval = S[ind_val,:]
    return Strn,Sval

def funcCVclustered(S,rmin,rmax,thresh):
    n,p = S.shape
    clusters = getYclusters(S,thresh)
    numclst = len(np.unique(clusters))
    all_ind = np.linspace(0,n-1,n)
    res = np.zeros((numclst,rmax-rmin+1))
    for ii in range(rmax-rmin+1):
        r = rmin+ii
        clst = 0
        while clst < numclst:
            clst_inds_logic = np.array(clusters==(clst+1)) # all cluster indices
            Sval = S[~clst_inds_logic,:] # validation array
            list_trn = []
            for jj in range(numclst):
                if (jj+1) != (clst+1):
                    current_clst_logic = np.array(clusters==(jj+1))
                    current_clst_inds = all_ind[current_clst_logic]
                    trn_ind = int(np.random.choice(current_clst_inds,size=1)[0])
                    list_trn.append(trn_ind)
            Strn = S[list_trn,:]
            try:
                Ar = getAr(Strn,r)
                Shat  = getShat(Ar,Sval)
                res[clst,ii] = np.linalg.norm(Sval[:,1:]-Shat[:,1:])/np.linalg.norm(Sval[:,1:])
            except Exception as e:
                print(getattr(e, 'message', repr(e)))
                print('r',r,'removal of cluster',clst+1,'gives error, trying next y-cluster')
            clst = clst+1
    return res

# finding optimal degree r by CV
cthresh = 0.02 # threshold
clusters = getYclusters(S,cthresh)
numclst = len(np.unique(clusters))
print('y-data has',numclst,'clusters')
rmin = 2
rmax = numclst-1
print('minimal degree n =',rmin)
print('maximal degree n =',rmax)
print('finding optimal degree by clustered CV')
res = funcCVclustered(S,rmin,rmax,cthresh)
resCV = np.mean(res,axis=0)
r = rmin+np.argmin(resCV)
print('optimal degree n =',r)

# training optimal-degree model
print('training optimal-degree model')
V = np.vander(y[:,0],r,increasing=True)
VtV = np.matmul(np.transpose(V),V)
print('cond(VtV) =',np.linalg.cond(VtV)) # Note: High r may lead to ill-conditioned VtV; stay within r < numclst

Ahat = np.linalg.solve(VtV,np.matmul(np.transpose(V),X))
AAt = np.matmul(Ahat,np.transpose(Ahat))
GammaXHat = np.linalg.solve(AAt,Ahat)
BetaXHat = np.matmul(np.transpose(Ahat),GammaXHat)
Xhat = np.matmul(X,BetaXHat)

XTrainRes = np.sum(np.abs(X-Xhat)**2,axis=0)/np.sum(np.abs(X)**2,axis=0)
XTrainErr = np.linalg.norm(X-Xhat)/np.linalg.norm(X)

A0hat = np.linalg.solve(VtV,np.matmul(np.transpose(V),y))
GammaYHat = np.linalg.solve(AAt,A0hat)
BetaYHat = np.matmul(np.transpose(Ahat),GammaYHat)
yhat = np.matmul(X,BetaYHat)
yTrainErr = np.linalg.norm(y-yhat)/np.linalg.norm(y)
print('X train (projection) error =',XTrainErr)
print('y train (projection) error =',yTrainErr)

# testing optimal-degree model
print('testing optimal-degree model')
nT,pT = dfTest.shape
pT = pT-3
datasetTest = dfTest.to_numpy()
XTest = datasetTest[:,1:pT+1]
yTest = np.zeros((nT,1))
yTest[:,0] = np.copy(datasetTest[:,-2])
# normalizing y-test by y-train scale parameters
yTest = yTest - ymean
yTest = yTest/ymax
STest = np.concatenate((yTest,XTest),axis=1)

XTestHat = np.matmul(XTest,BetaXHat)
XTestRes = np.sum(np.abs(XTest-XTestHat)**2,axis=0)/np.sum(np.abs(XTest)**2,axis=0)
XTestErr = np.linalg.norm(XTest-XTestHat)/np.linalg.norm(XTest)

yTestHat = np.matmul(XTest,BetaYHat)
yTestRes = np.abs(yTest-yTestHat)**2/np.abs(yTest)**2
yTestErr = np.linalg.norm(yTest-yTestHat)/np.linalg.norm(yTest)

print('testing samples, nT =',nT)
print('X test error =',XTestErr)
print('y test error =',yTestErr)

print('filtering features')
# finding optimal threshold for feature selection
numthr = 200
thrmin = np.min(XTestRes)+1e-3*(np.max(XTestRes)-np.min(XTestRes))
thrmax = np.max(XTestRes)
thrArr = np.logspace(np.log10(thrmin),np.log10(thrmax),num=numthr,endpoint=True)
XFTrainErrArr = []
XFTestErrArr = []
yFTestErrArr = []
yFTrainErrArr = []

for ii in range(numthr):
    thr = thrArr[ii]
    BinFilter = np.array(XTestRes<=thr)
    XF = np.copy(X[:,BinFilter])
    XFTest = np.copy(XTest[:,BinFilter])
    AFhat = np.linalg.solve(VtV,np.matmul(np.transpose(V),XF))
    # make sure that a sufficient number of features is remaining!!!
    AFAFt = np.matmul(AFhat,np.transpose(AFhat))
    GammaXFHat = np.linalg.solve(AFAFt,AFhat)
    BetaXFHat = np.matmul(np.transpose(AFhat),GammaXFHat)

    #XFhat = np.matmul(V[:,0:r],AFhat[0:r,:])
    XFhat = np.matmul(XF,BetaXFHat)
    XFTrainErr = np.linalg.norm(XF-XFhat)/np.linalg.norm(XF)

    XFTestHat = np.matmul(XFTest,BetaXFHat)
    XFTestErr = np.linalg.norm(XFTest-XFTestHat)/np.linalg.norm(XFTest)

    GammayFHat = np.linalg.solve(AFAFt,A0hat)
    BetayFHat = np.matmul(np.transpose(AFhat),GammayFHat)
    yFTrainHat = np.matmul(XF,BetayFHat)
    yFTrainErr = np.linalg.norm(y-yFTrainHat)/np.linalg.norm(y)

    yFTestHat = np.matmul(XFTest,BetayFHat)
    yFTestErr = np.linalg.norm(yTest-yFTestHat)/np.linalg.norm(yTest)

    XFTrainErrArr.append(XFTrainErr)
    XFTestErrArr.append(XFTestErr)
    yFTestErrArr.append(yFTestErr)
    yFTrainErrArr.append(yFTrainErr)

thropt = thrArr[np.argmin(yFTrainErrArr)]
BinFilter = np.array(XTestRes<=thropt)
XF = np.copy(X[:,BinFilter])
XFTest = np.copy(XTest[:,BinFilter])
XFTestErr = np.linalg.norm(XTest[:,BinFilter]-XTestHat[:,BinFilter])/np.linalg.norm(XTest[:,BinFilter])

AFhat = np.linalg.solve(VtV,np.matmul(np.transpose(V),XF))
AFAFt = np.matmul(AFhat,np.transpose(AFhat))
GammayFHat = np.linalg.solve(AFAFt,A0hat)
BetayFHat = np.matmul(np.transpose(AFhat),GammayFHat)
yFTestHat = np.matmul(XFTest,BetayFHat)
yFTestErr = np.linalg.norm(yTest-yFTestHat)/np.linalg.norm(yTest)

print('features remaining after filtering:',np.sum(BinFilter),'out of',p)
print('X test error after filtering =',XFTestErr)
print('y test error after filtering =',yFTestErr)

# plots-------------------------------------------------------------------------
plt.ion()

yu = np.linspace(np.min(y),np.max(y),n)
Vu = np.vander(yu,r,increasing=True)
Xu = np.matmul(Vu,Ahat)

plt.figure(2,dpi=dpi)
plt.clf()
fig2,ax = plt.subplots(nrows=1,ncols=2,sharey=True,num=2)

colors = ['tab:blue','tab:orange','tab:green','tab:cyan','tab:brown','tab:purple']
features = [15,29,53,100,178,229]
cc = 0
for w in features:
    col = colors[cc]
    if BinFilter[w]:
        linestyle='-'
        pp = 0
    else:
        pp = 1
        linestyle='--'
    ax[pp].plot(y,X[:,w],'o',color=col,label='col. '+str(w)+' train')
    ax[pp].plot(yu,Xu[:,w],linestyle,color=col)
    ax[pp].plot(yTest,XTest[:,w],'d',color=col,label='col. '+str(w)+' test')
    cc = cc+1

ax[0].set_xlabel(r'$y$')
ax[1].set_xlabel(r'$y$')
ax[0].set_ylabel(r'$x_{j}(y)$')
ax[0].set_title('Accepted columns')
ax[1].set_title('Rejected columns')
fig2.savefig('Yarn_figure2.pdf')

plt.figure(3,dpi=dpi)
plt.clf()
fig3,ax = plt.subplots(nrows=3,ncols=1,sharex=True,num=3)
ax[0].semilogy(XTestRes.transpose(),label=r'$\chi_j$')
ax[0].semilogy(np.linspace(0,pT-1,pT),np.ones(pT)*thropt,'--',color='tab:red',label=r'$\tau_{opt}$')
ax[0].legend(loc='center right')
ax[0].set_ylabel(r'$\chi_{j}$')
ax[1].plot(X.transpose())
ax[1].set_ylabel(r'$X$-data')
xmina0,xmaxa0,ymina0,ymaxa0 = ax[0].axis()
xmina1,xmaxa1,ymina1,ymaxa1 = ax[1].axis()
cc = 0
for jj in range(p):
    if ~BinFilter[jj]:
        ax[1].plot([jj,jj],[ymina1,ymaxa1],color='tab:gray',linewidth=3,alpha=0.3)
    if jj in features:
        if BinFilter[jj]:
            linestyle='-'
            col = colors[cc]
        else:
            linestyle='--'
            col = colors[cc]
        ax[0].plot([jj,jj],[ymina0,ymaxa0],linestyle,color=col,linewidth=2,alpha=1)
        ax[1].plot([jj,jj],[ymina1,ymaxa1],linestyle,color=col,linewidth=2,alpha=1)
        cc = cc+1

cc = 0
for nn in range(min(3,r)):
    ax[2].plot(Ahat[nn,:],label=r'$\hat{A}$['+str(nn)+',:]')
ax[2].grid('on')
ax[2].legend(loc='center right')
ax[2].set_ylabel(r'$\hat{A}$')
xmina2,xmaxa2,ymina2,ymaxa2 = ax[2].axis()
for jj in range(p):
    if ~BinFilter[jj]:
        ax[2].plot([jj,jj],[ymina2,ymaxa2],color='tab:gray',linewidth=3,alpha=0.3)
    if jj in features:
        if BinFilter[jj]:
            linestyle='-'
            col = colors[cc]
        else:
            linestyle='--'
            col = colors[cc]
        ax[2].plot([jj,jj],[ymina2,ymaxa2],linestyle,color=col,linewidth=2,alpha=1)
        cc = cc+1
fig3.savefig('Yarn_figure3.pdf')

plt.figure(4,dpi=dpi)
plt.clf()
fig4,ax = plt.subplots(nrows=1,ncols=2,num=4)
ax[0].plot(np.linspace(rmin,rmax,rmax-rmin+1),resCV,'-',color='tab:blue',label=r'$\langle\rho(X_{v})\rangle$')
ax[0].plot(r,resCV[r-rmin],'s',color='tab:red')
ax[0].set_xlabel(r'degree, $r$')
ax[0].legend()

ax[1].semilogx(thrArr,yFTrainErrArr,'-',color='tab:blue',label=r'$\rho(y)$')
ax[1].semilogx(thrArr,yFTestErrArr,'--',color='tab:gray',alpha=0.7,label=r'$\rho(y_{t})$')
ax[1].semilogx(thropt,yFTrainErrArr[np.argmin(yFTrainErrArr)],'s',color='tab:red')
ax[1].set_xlabel(r'threshold, $\tau$')
ax[1].legend()
fig4.savefig('Yarn_figure4.pdf')

plt.figure(5,dpi=dpi)
plt.clf()
fig5,ax = plt.subplots(nrows=1,ncols=1,num=5)
ax.plot(np.linspace(np.min(y*ymax+ymean),np.max(y*ymax+ymean),100),np.linspace(np.min(y*ymax+ymean),np.max(y*ymax+ymean),100),'-',color='tab:gray')
ax.plot(y*ymax+ymean,yhat*ymax+ymean,'o',color='tab:blue',label='training')
ax.plot(yTest*ymax+ymean,yTestHat*ymax+ymean,'d',color='tab:green',label=r'test, all predictors')
ax.plot(yTest*ymax+ymean,yFTestHat*ymax+ymean,'s',color='tab:red',label=r'test, improper predictors removed')
ax.set_aspect('equal', 'box')
ax.set_xlabel(r'measured $y$')
ax.set_ylabel(r'predicted $y$')
ax.legend(loc='upper left', bbox_to_anchor=(0,1.1))
fig5.savefig('Yarn_figure5.pdf')
