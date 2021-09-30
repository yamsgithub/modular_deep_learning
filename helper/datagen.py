#!/usr/bin/env python
import numpy,math
import numpy.random as nr
from matplotlib import pyplot 
import numpy.linalg as la

DEBUGPLOTS = False

class user_data:
  def __init__(self, Pc=lambda k: "black", Pm=lambda k: "x"):
    self.Px=[]
    self.Py=[]
    self.Pb=[]
    self.Pk=[]
    self.Pc=Pc
    self.Pm=Pm

  def onclick(self, event):
    self.Px.append(event.xdata)
    self.Py.append(event.ydata)
    self.Pb.append(event.button)
    self.Pk.append(event.key)
    self.ax.scatter(self.Px[-1],self.Py[-1],c=self.Pc(self.Pk[-1]),marker=self.Pm(self.Pk[-1]))
    pyplot.draw()

  def show(self):
    self.fig, self.ax = pyplot.subplots(1,1,figsize=(8,8))
    self.ax.set_xlim(-.1,1.1)
    self.ax.set_ylim(-.1,1.1)
    #self.ax.scatter([.5],[.5],c="black",marker="x")
    self.ax.scatter(self.Px,self.Py,c="black",marker="x")
    self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
    if __name__=="__main__":pyplot.show()
  
  def get_n(self):
    return len(self.Px)
  
  def get_instances(self):
    return numpy.array(self.Px).reshape(len(self.Px),1)

  def get_labels(self):
    return numpy.array(self.Py)
    
class user_data_c(user_data):
  def __init__(self):
    super().__init__(Pc=lambda k: "blue" if k=="shift" else "orange",Pm=lambda k: "o")

  def get_n(self):
    return len(self.Px)
  
  def get_instances(self):
    return numpy.array((self.Px,self.Py)).T

  def get_labels(self):
    return numpy.array([+1 if k=="shift" else -1 for k in self.Pk])

class sin_data(user_data):
  def __init__(self,freqs=[3,5,7,13,23],n=50,noise=.05,seed=None):
    self.freqs=nr.randint(1,50,5) if freqs=="random" else freqs
    print(self.freqs)
    self.compute= lambda xs: numpy.sum([numpy.sin(f*xs) for f in self.freqs],axis=0) / len(self.freqs)
    self.seed=nr.randint(2**32)  if seed == None  else seed
    print(self.seed)
    self.rs=nr.RandomState(self.seed)
    self.Px=self.rs.random_sample(n)
    self.noise= lambda num: self.rs.normal(scale=noise,size=num)
    self.Py=self.compute(self.Px) + self.noise(n)

  def show(self, truefunc=True):
    self.fig, self.ax = pyplot.subplots(1,1,figsize=(8,8))
    #self.ax.set_xlim(-.1,1.1)
    #self.ax.set_ylim(-.1,1.1)
    self.ax.scatter(self.Px,self.Py,c="black",marker="x")
    xs=numpy.linspace(0,1,200)
    ys=self.compute(xs)
    if truefunc: self.ax.plot(xs,ys)
    #self.ax.scatter(self.Px,self.Py,c="black",marker="x")
    if __name__=="__main__":pyplot.show()
  
  def evaluate(self,ls,s=.1,res=120,K=None):
    EmpRMSEs=[]
    ExpRMSEs=[]
    noise=self.noise(res)
    Kfun=(lambda s,D,K: numpy.exp(-D/(2*s**2))) if type(K)==type(None) else K 
    n=len(self.Px)
    X=numpy.array((self.Px),ndmin=2).T
    Y=numpy.array(self.Py)
    Klin=numpy.dot(X,X.T)
    d=numpy.diag(Klin).reshape(1,n)
    D=-2*Klin+d+d.T
    Kact=Kfun(s,D,Klin)
    ZT=numpy.linspace(-0,1.0,res).reshape(1,res)
    KXZlin=numpy.dot(X,ZT)
    KZlin=numpy.dot(ZT.T,ZT)
    dZ=numpy.diag(KZlin).reshape(1,res)
    DXZ=-2*KXZlin + d.T + dZ
    YZ=self.compute(ZT)+noise
    for l in ls:
      c= la.solve( Kact + l*numpy.diag(numpy.ones(n)), Y )
      Yhat=numpy.dot(Kact,c)
      EmpRMSEs.append(math.sqrt(numpy.dot(Y-Yhat,Y-Yhat)/n))
      PZ=numpy.dot(Kfun(s,DXZ.T,KZlin.T),c)
      diff=(YZ-PZ).reshape(res)
      ExpRMSEs.append(math.sqrt(numpy.dot(diff,diff)/res))
    return (EmpRMSEs,ExpRMSEs)
        
    
class gm_kmc(user_data_c):
  def __init__(self,ininum=1000,nn=0.05,clu=5,nump=100,noise=.05,seed=None,labeler=lambda l:[i%2 *2 -1 for i in l],col=lambda k: "blue" if k>0 else "orange"):#nn=0.05
    super()
    self.seed=nr.randint(2**32)  if seed == None  else seed
    print(self.seed)
    self.rs=nr.RandomState(self.seed)
    self.nn=nn
    no=numpy.sqrt(self.rs.uniform(0,1,ininum))
    a=self.rs.uniform(0,2*numpy.pi,ininum)
    self.y=numpy.sin(a)*no
    self.x=numpy.cos(a)*no
    if DEBUGPLOTS: 
        self.fig, self.ax = pyplot.subplots(1,1,figsize=(8,8))
        pyplot.scatter(self.x,self.y, alpha=.5)
    self.rawX=numpy.array((self.x,self.y)).T
    Klin=numpy.dot(self.rawX,self.rawX.T)
    d=numpy.diag(Klin).reshape(1,ininum)
    self.D=-2*Klin+d+d.T
    self.lm=[self.rs.randint(ininum)]
    if DEBUGPLOTS: self.ax.annotate(str(len(self.lm)), (self.x[self.lm[-1]], self.y[self.lm[-1]]))
    self.addkmcclu(clu-1)
    self.nump=nump
    (data,cluster)=self.get_points(nump=nump)
    self.data=data
    self.cluster=cluster
    self.labelisnoisy=[(self.rs.uniform()<noise) for i in cluster]
    #print(self.cluster, self.labelisnoisy)
    self.labeler=labeler
    self.noise=noise
    self.col=col
    #self.Px=data[:,0]    
    #self.Py=data[:,1]
    #self.Pk=["shift" if (i%2)==(self.rs.uniform()<noise)*1 else None for i in cluster]

  def addkmcclu(self, clu):
    for i in range(clu):
      dl=numpy.min(self.D[self.lm,:]**2,axis=0) 
      cdl=numpy.cumsum(dl)
      r=self.rs.random()*cdl[-1]
      geq=numpy.flatnonzero(cdl>=r) 
      lt=numpy.flatnonzero(cdl<r)  # nonzero won't work... 
      self.lm+=[numpy.min(geq) if len(geq)>0 else 0]
      if DEBUGPLOTS: self.ax.annotate(str(len(self.lm)), (self.x[self.lm[-1]], self.y[self.lm[-1]]))   
    self.sigmas=self.nn*numpy.min((self.D[self.lm,:][:,self.lm])[self.D[self.lm,:][:,self.lm]>0].reshape((len(self.lm),len(self.lm)-1)),axis=1)

  def get_n(self):
    return len(self.cluster)
  
  def get_instances(self):
    return self.data

  def get_true_labels(self):
    return self.labeler(self.cluster)

  def get_labels(self):
    return [i*(-2*r+1) for i,r in zip(self.get_true_labels(),self.labelisnoisy)]


  def get_points(self,nump=100):
    Xclus=[]
    Yclus=[]
    I = numpy.eye(2)
    clu=len(self.lm)
    for i in range(clu):
        Xclu=self.rs.multivariate_normal(self.rawX[self.lm[i],:],self.sigmas[i]*I,nump if type(nump)==int else nump[i])
        Xclus.append(Xclu)
        if DEBUGPLOTS: self.ax.scatter(Xclu[:,0],Xclu[:,1],alpha=.3)
        Yclus.extend([i]*(nump if type(nump)==int else nump[i]))
    return (numpy.vstack(Xclus)+1)/2, Yclus
  
  def scatter(self, ax):
    ax.scatter(self.data[:,0],self.data[:,1],c=[self.col(k) for k in self.get_labels()],alpha=.4)

  def show(self):
    #print(self.data.shape,self.get_labels())
    self.fig, self.ax = pyplot.subplots(1,1,figsize=(8,8))
    self.scatter(self.ax)
    #self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
    

  
  def evaluate(self,ls,s=.1,res=1000,K=None,rtncoeffinfo=False):
    from qpsolvers import solve_qp
    EmpRMSEs=[]
    ExpRMSEs=[]
    ubcoeffs=[]
    lbcoeffs=[]
    minnzcs=[]
    Kfun=(lambda s,D,K: numpy.exp(-D/(2*s**2))) if type(K)==type(None) else K 
    n=self.get_n()#len(self.Px)
    X=self.get_instances()#numpy.array((self.Px,self.Py),ndmin=2).T
    print(X.shape)
    Y=self.get_labels()#numpy.array([+1 if k=="shift" else -1 for k in self.Pk])
    Klin=numpy.dot(X,X.T)
    d=numpy.diag(Klin).reshape(1,n)
    D=-2*Klin+d+d.T
    Kact=Kfun(s,D,Klin)    
    Kx=Kfun(s,D,K)
    Q=numpy.dot(numpy.dot(numpy.diag(Y),Kx+.0000001*numpy.diag(numpy.ones(n))),numpy.diag(Y))
    Z, Zc = self.get_points(res)
    ZL=[(i%2)*2-1 for i in Zc]
    print(Z.shape)
    ZLn=numpy.array([i if self.rs.uniform()<self.noise else -i for i in ZL])
    KXZlin=numpy.dot(X,Z.T)
    KZlin=numpy.dot(Z,Z.T)
    tp=Z.shape[0]
    dZ=numpy.diag(KZlin).reshape(1,tp)
    DXZ=-2*KXZlin + d.T + dZ
    Kz=Kfun(s,DXZ,KZlin)
    for l in ls:
      c=solve_qp(Q+.0000001*numpy.diag(numpy.ones(n)), -numpy.ones(n), lb=numpy.zeros(n),ub=numpy.ones(n)/l)
      PX=numpy.dot(c,numpy.dot(numpy.diag(Y),Kx))#.reshape(res,res)
      PZ=numpy.dot(c,numpy.dot(numpy.diag(Y),Kz))#.reshape(res,res)
      EmpRMSEs.append(numpy.sum(PX*Y<0)/n)
      ExpRMSEs.append(numpy.sum(PZ*ZLn<0)/tp)
      ubcoeffs.append(numpy.sum(c>.99*1./l))
      lbcoeffs.append(numpy.sum(c<.0001*1./l))
      minnzcs.append(numpy.min(c[c>0]))
    return (EmpRMSEs,ExpRMSEs) if not rtncoeffinfo else (EmpRMSEs,ExpRMSEs,ubcoeffs,lbcoeffs,minnzcs) 

if __name__=="__main__":
    DEBUG=True 
    DEBUGPLOTS=True 
    if DEBUG:
      #o=gm_kmc()
      o=sin_data()
      o.show()
      print(o.evaluate([.1]))
      
      o=user_data("random")
      o.show()
      print(o.Px)
      print(o.Py)
      print(o.Pb)
      print(o.Pk)
      
      o=user_data_c()#(Pc=lambda k: "blue" if k=="shift" else "orange",Pm=lambda k: "o")
      o.show()
