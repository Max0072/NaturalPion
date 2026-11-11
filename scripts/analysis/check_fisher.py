import numpy as np
rng=np.random.default_rng(0); m,n,N=5,4,400000
W=rng.standard_normal((m,n))
La=rng.standard_normal((n,n)); Ld=rng.standard_normal((m,m))
x=rng.standard_normal((N,n))@La.T; d=rng.standard_normal((N,m))@Ld.T
A=La@La.T; D=Ld@Ld.T; S=W.T@D@W; V=W@A@W.T
sk=lambda M:(M-M.T)/2
X=sk(rng.standard_normal((n,n))); Y=sk(rng.standard_normal((m,m)))
u=d@W; v=x@W.T
inner=np.einsum('bi,ij,bj->b',u,X,x)
F_mc=sk(np.einsum('b,bi,bj->ij',inner,u,x)/N); F_th=0.5*(S@X@A+A@X@S)
print('in  rel err',np.abs(F_mc-F_th).max()/np.abs(F_th).max())
print('pullback  ',np.abs(sk(W.T@D@W@X@A)-F_th).max())
inner=np.einsum('bi,ij,bj->b',d,Y,v)
F_mc=sk(np.einsum('b,bi,bj->ij',inner,d,v)/N); F_th=0.5*(D@Y@V+V@Y@D)
print('out rel err',np.abs(F_mc-F_th).max()/np.abs(F_th).max())
wa,Ua=np.linalg.eigh(A); Ah=Ua@np.diag(wa**-0.5)@Ua.T
l,Q=np.linalg.eigh(Ah@S@Ah); P=Ah@Q
G=sk(rng.standard_normal((n,n)))
Yb=(P.T@G@P)/(0.5*(l[:,None]+l[None,:])); Xs=P@Yb@P.T
print('inverse res',np.abs(0.5*(S@Xs@A+A@Xs@S)-G).max(),'skew',np.abs(Xs+Xs.T).max())
c=0.7/np.linalg.norm(Xs,2); K=c/2*Xs; R=np.linalg.solve(np.eye(n)+K,np.eye(n)-K)
print('angle',np.abs(np.angle(np.linalg.eigvals(R))).max(),2*np.arctan(0.35))
print('orth',np.abs(R.T@R-np.eye(n)).max())
