
import numpy as num
import scipy as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D
from sklearn import manifold, datasets
#"definition of inputs. will be deleted later
#y=num.matrix([[1, 2],[3, 4]]);
#y=num.matrix([[1, 2, 5, 12, 4, 2, 6, 2],[3, 4, 3, 1, 7, 4, 8, 4],[ 5, 6, 7, 3, 4, 2, 72, 1]])
#y, color = datasets.samples_generator.make_swiss_roll(n_samples=400)
y=num.asmatrix(Y);
#how many neighbors
k=9;
#Reduced dimension
dx=2;

#functions

#Make Neibourhood Graph from y
def NGraph(Y, k):
    y2sum=num.sum(num.power(y,2),0);
    #Dmat is distance matrix
    Dmat=num.sqrt(y2sum.T+y2sum-2*(Y.T*Y));
    #Sort Dmat to know what is sorted distance to each point
    V=num.sort(Dmat,0);
    #remain just k nearest neighbor
    G=(Dmat<=num.matlib.repmat(V[k,:],n,1))-num.eye(n);
    G=1*(G+G.T>=1);
    return G, Dmat

#Compute Ltilde
def compute_Ltilde(L,epsilon,gamma,opt_dec):
    n=num.ma.size(L,0);
    if (opt_dec==0):
        result=num.linalg.inv(epsilon*num.ones([n,1])*num.ones([1,n])+2*gamma*L);
        return result;
    else:
        (U_L, D_L, V_Lh)=num.linalg.svd(L);
        V_L=V_Lh.T;
    
    Depsilon_inv=num.zeros([n,n]);
    #Don't flip sign of singular vectors
    sign_sin_val=(V_L[:,-1].T)*U_L[:,-1];
    Depsilon_inv[n-1,n-1]=sign_sin_val/(epsilon*n);
    Ltilde_epsilon=V_L*Depsilon_inv*(U_L.T);
    
    D_L_inv=num.zeros([n,n]);
    D_L_inv[0:n-1,0:n-1]=num.diagflat(1/(D_L[0:n-1]));        
    Ltilde_L=V_L*D_L_inv*(U_L.T);
    
    #Check if they match
    Ltilde= Ltilde_epsilon+1/(2*gamma)*Ltilde_L;
    return Ltilde, Ltilde_epsilon, Ltilde_L, D_L.T;

#Compute A
def findA(Ltilde,G, ECC_rs, gamma):
    
    UTV=ECC_rs;
    B=G-num.diagflat(num.sum(G,0));
    
    UTVG=UTV*G;
    LTLG=Ltilde*G;
    GTUTV=G.T*UTV;
    GTL2=G.T*Ltilde;
    
    line1=(B.T)*((num.multiply(Ltilde,UTV))*B-(num.multiply(Ltilde,UTVG))+(num.multiply(LTLG,UTV)));
    line2=-(num.multiply(Ltilde,GTUTV))*B+(num.multiply(Ltilde,(G.T)*UTVG))-(num.multiply(GTUTV,LTLG));
    line3=(num.multiply(GTL2,UTV))*B-(num.multiply(GTL2,UTVG))+(num.multiply(UTV,(G.T)*LTLG));
    
    return (gamma**2)*(line1+line2+line3);

#Compute Gamma
def findGamma(Ltilde,G, EXX_rs):
    UTV=EXX_rs;
    B=G+num.diagflat(num.sum(G,0));
    
    UTVG=UTV*G;
    LTLG=Ltilde*G;
    GTUTV=G.T*UTV;
    GTL2=G.T*Ltilde;
    
    line1=(B.T)*((num.multiply(Ltilde,UTV))*B-(num.multiply(Ltilde,UTVG))-(num.multiply(LTLG,UTV)));
    line2=-(num.multiply(Ltilde,GTUTV))*B+(num.multiply(Ltilde,(G.T)*UTVG))+(num.multiply(GTUTV,LTLG));
    line3=-(num.multiply(GTL2,UTV))*B+(num.multiply(GTL2,UTVG))+(num.multiply(UTV,(G.T)*LTLG));
    
    return (line1+line2+line3);
#Compute Gamma SVD
def compute_Gamma_svd(G, EXX, y,gamma,Ltilde_epsilon,Ltilde_L):
    #Parameters
    n=num.ma.size(y,1);
    dx=int(num.ma.size(EXX,1)/n);
    
    #Find Gamma
    AllCoeff=num.zeros([n,n,dx,dx]);
    for r in range(0,dx):
        for s in range(0,dx):
            EXX_rs=EXX[r::dx,s::dx];
            Coeff_rs=findGamma(Ltilde_L,G, EXX_rs);
            AllCoeff[:,:,r,s]=Coeff_rs;
    AllCoeff=num.transpose(AllCoeff,[2,3,0,1]);
    AllCoeff=num.transpose(AllCoeff,[0,2,1,3]);
    Gam_L=num.reshape(AllCoeff,[dx*n,dx*n],order='F');
    Gam=(gamma/2)*Gam_L;
    return Gam,Gam_L;
    
    
#Compute A, b
def findAandb(G, mean_c, cov_c, y, gamma, epsilon):
    
    #Parameters needed
    dy=num.ma.size(y,0);
    n=num.ma.size(y,1);
    dx=int(num.ma.size(mean_c,1)/n);
    
    h=num.sum(G,1);
    
    #Computing b:
    prodm=(mean_c.T)*y;
    rowblock=num.split(num.multiply(num.kron(G,num.ones([dx,1])),prodm),n); #Check it again
    b1=num.reshape(num.asmatrix(num.sum(rowblock,0)),[dx*n,1],order='F'); #check if asmatrix doing well
    b2=-num.sum(((num.multiply(prodm.T,num.kron(num.eye(n),num.ones([1,dx]))))*num.kron(G,num.eye(dx))),0).T;
    b3=-(num.sum(num.multiply(num.kron(G,num.ones([dx,1])),prodm),1));
    b4=num.multiply(num.kron(h,num.ones([dx,1])),num.sum(num.multiply(num.kron(num.eye(n),num.ones([dx,1])),prodm),1));
    b_without_gamma=b1+b2+b3+b4;
    b=gamma*b_without_gamma;
    
    #Computing A
    L=num.diagflat(h)-G;
    Ltilde=num.linalg.inv(epsilon*num.ones([n,n])+2*gamma*L);
    #E[C^T C]
    ECTC=dy*cov_c+(mean_c.T)*mean_c;
    
    AllCoeff=num.zeros([n,n,dx,dx]);
    for r in range(0,dx):
        for s in range(0,dx):
            ECC_rs=ECTC[r::dx,s::dx];
            Coeff_rs=findA(Ltilde,G, ECC_rs, gamma);
            AllCoeff[:,:,r,s]=Coeff_rs;
    AllCoeff=num.transpose(AllCoeff,[2,3,0,1]);
    AllCoeff=num.transpose(AllCoeff,[0,2,1,3]);
    A=num.reshape(AllCoeff,[dx*n,dx*n],order='F');
    return A, b;

def findGammaandh(G, mean_x, cov_x, y, gamma, Ltilde_epsilon, Ltilde_L):
    
    #Parameters
    dy=num.ma.size(y,0);
    n=num.ma.size(y,1);
    dx=int(num.size(cov_x,1)/n);
    
    #find h?
    h=num.sum(G,1);
    
    #Compute H
    kronGones=num.kron(G,num.ones([1,dx]));
    kronINones=num.kron(num.eye(n),num.ones([1,dx]))
    
    Yd=num.reshape(y,[dy,n],order='F');
    Xd=num.reshape(mean_x,[dx,n],order='F');
    e1=num.multiply(num.matlib.repmat(Xd,n,1).T,kronGones)-num.multiply(num.matlib.repmat(num.reshape((Xd*G),[n*dx,1],order='F').T,n,1),kronINones);
    e2=(Yd*G)*(num.multiply(num.matlib.repmat(num.reshape(Xd,[n*dx,1],order='F').T,n,1),kronINones));
    e3=(num.multiply(num.matlib.repmat(h.T,dy,1),Yd))*(num.multiply(kronINones,num.matlib.repmat(mean_x.T,n,1)));
    H=Yd*e1-e2+e3;
    
    #Computing Gamma
    EXX=cov_x+mean_x*(mean_x.T);
    (Gamma, Gamma_L)=compute_Gamma_svd(G,EXX,y,gamma,Ltilde_epsilon,Ltilde_L);
    return Gamma, H, Gamma_L;

#Update gamma
def exp_log_likeli_update_gamma(mean_c, cov_c, H, y, L, epsilon, eigL, QhatLtilde_LQhat):
    dy=num.ma.size(mean_c,0);
    n=num.ma.size(L,0);
    
    secondmoment= dy*cov_c+(mean_c.T)*mean_c;
    
    QhatLtilde_LQhatsecondmoment= num.trace(QhatLtilde_LQhat*secondmoment);
    mean_cH=num.trace((mean_c.T)*H);
    Lyy=num.trace(L*(y.T)*y);
    #find gamma as in equation ...
    gamma=-0.5*dy*(n-1)/(-0.25*QhatLtilde_LQhatsecondmoment+mean_cH-Lyy);
    
    l1=-gamma/4*QhatLtilde_LQhatsecondmoment;
    l2=gamma*mean_cH;
    l3=-0.5*num.trace(epsilon*num.ones([n,1])*num.ones([1,n])*(y.T)*y)-gamma*Lyy;
    l4=-0.5*n*dy*num.log(2*num.pi)+0.5*dy*num.sum(num.log(eigL[0:n-1]))+0.5*dy*num.log(n*epsilon)+0.5*dy*(n-1)*num.log(2*gamma);
    return l1+l2+l3+l4, gamma;

#Log(Det(.))
def logdet(A):
    P, L, U=sci.linalg.lu(A);
    return num.sum(num.log(num.abs(num.diag(L))))+num.sum(num.log(num.abs(num.diag(U))));
#-Dkl(q(c)||p(C|G,\theta))
def DklC(mean_c, cov_c, invOmega, J, epsilon):
    #parameters
    dy=num.ma.size(mean_c,0);
    ndx=num.ma.size(cov_c,0);
    
    epJJinvOmega=epsilon*J*(J.T)+invOmega;
    covCepJJinvOmega=cov_c*epJJinvOmega;
    
    logdetepJJ=logdet(epJJinvOmega);
    logdetcovc=logdet(cov_c);
    logdettrm=logdetepJJ+logdetcovc;
    
    lwb_C=0.5*dy*logdettrm-0.5*dy*num.trace(covCepJJinvOmega)+0.5*ndx*dy-0.5*num.trace(epJJinvOmega*(mean_c.T)*mean_c);
    return lwb_C

def Dklalpha(mean_x, cov_x, invOmega, eigv_L)    :
    
    #parameters
    ndx=num.ma.size(mean_x,0);
    dx=int(ndx/(num.ma.size(eigv_L,0)));
    
    invOmegaQuad=(mean_x.T)*invOmega*mean_x;
    mean_x2=(mean_x.T)*mean_x;
    tr_cov_x=num.ma.trace(cov_x);
    
    alpha=0.001;
    
    logdettrm=logdet(cov_x);
    
    lwb_x=0.5*logdettrm+0.5*dx*num.sum(num.log(alpha+2*eigv_L))-0.5*alpha*tr_cov_x-0.5*num.trace(invOmega*cov_x)+0.5*ndx-0.5*alpha*mean_x2-0.5*invOmegaQuad;
    return lwb_x, alpha;

    
#"Start of code


#Size
    
dy=num.ma.size(y,0);
n=num.ma.size(y,1);

#Find Neighborhood Graph
(G, Dmat)=NGraph(y,k)
#Center Data
mm=num.matlib.mean(y,1);
y=y-mm;
#divide data
mm=num.max(num.abs(y[:]));
y=y/mm;


#make laplacian
L=num.diagflat(num.sum(G,0))-G;

#Create \omega^-1
invOmega=num.matlib.kron(2*L,num.eye(dx));

#initializer of alpha
alpha0=1;

#Maximum iteration
max_em_iter=100;

#% invPi is Pi^-1 where p(x|G, alpha) = N(x | 0, Pi) (prior of X).
#init of invPi
invPi=alpha0*num.eye(n*dx)+invOmega;

#initializer of gamma
gamma0=1;

#define epsilon
epsilon=0.1;

J=num.asmatrix(num.kron(num.ones([n,1]),num.eye(dx)));
#initial Value of Cov_c
cov_c0=num.linalg.inv(epsilon*(J*(J.T))+invOmega) ;

#initial value of mean_c
#mean_c0=num.matlib.randn(dy,dx*n)*cov_c0.T;
#change it to former state
mean_c0=num.matlib.ones([dy,dx*n])*cov_c0.T;

#Iterations
abs_tol=0.0001;
#Start EM

gamma_new=gamma0;
invPi_new=invPi;

opt_dec=1; #Using decomposition
(Ltilde, Ltilde_epsilon, Ltilde_L, eigv_L)=compute_Ltilde(L,epsilon,gamma_new,opt_dec);


#update Cov_c
cov_c=cov_c0;
mean_c=mean_c0;


#Matrix of covariance and mean of distributions
meanCmat=num.zeros([n*dx*dy,max_em_iter]);
meanXmat=num.zeros([n*dx,max_em_iter]);
covCmat=num.zeros([n*dx,max_em_iter]);
covXmat=num.zeros([n*dx,max_em_iter]);

alphamat=num.zeros([max_em_iter,1]);
gammamat=num.zeros([max_em_iter,1]);

#current lower bound
prev_lwb=num.inf;

#lower bound in any iteration
lwbs=num.zeros([max_em_iter,1]);

for i_em in range(0,max_em_iter):
    
    #E Step
    #Computation of q(x)
    (A, b)= findAandb(G, mean_c, cov_c, y, gamma_new, epsilon);#Check if A is right or not. I'm not sure aboute dimension transpose
    cov_x=num.linalg.inv(A+invPi_new);
    mean_x=cov_x*b;
    
    #Computation of q(C)
    (Gamma, H, Gamma_L)=findGammaandh(G,mean_x,cov_x,y,gamma_new,Ltilde_epsilon,Ltilde_L);
    cov_c=num.linalg.inv(Gamma+epsilon*(J)*(J.T)+invOmega);
    mean_c=gamma_new*H*(cov_c.T);
    
    
    #M Step
    (lwb_likelihood, gamma_new)=exp_log_likeli_update_gamma(mean_c, cov_c, H, y, L, epsilon, eigv_L, Gamma_L);
    lwb_C=DklC(mean_c,cov_c, invOmega, J, epsilon);
    (lwb_x, alpha_new)=Dklalpha(mean_x, cov_x, invOmega, eigv_L);
    
    #Update InvPi
    
    invPi_new=alpha_new*num.eye(n*dx)+invOmega;
    
    #lower bound
    lwb=lwb_likelihood+lwb_C+lwb_x;
    
    lwbs[i_em]=lwb;
    
    #save
    meanCmat[:,i_em]=num.reshape(mean_c,[1,n*dx*dy],order='F');
    meanXmat[:,i_em]=num.reshape(mean_x,[1,n*dx]);
    covCmat[:,i_em]=num.diag(cov_c);
    covXmat[:,i_em]=num.diag(cov_x);
    
    alphamat[i_em]=alpha_new;
    gammamat[i_em]=gamma_new;
    print("Iteration",i_em)
    if (i_em>=2 and num.abs(lwb-prev_lwb)<abs_tol):
        break;
    prev_lwb=lwb;
 # End of the function


