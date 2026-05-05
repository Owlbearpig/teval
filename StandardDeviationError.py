# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:14:47 2019

@author: machc
"""

import numpy as np

def TransferfunctionError(Esample,Eref,Esamcorrf,Erefcorrf,noiseposition):
    #H=Esam/Eref*1/Fcor
    #Fcor=(Esamcorr/Erefcorr)
    #H=Esam*Erefcorr/(Eref*Esamcorr)=((a+ib)*(m+in))/((c+id)*(k+il))
    #f=Re(H)
    #g=Im(H)
    a=np.mean(Esample.real,0)
    b=np.mean(Esample.imag,0)
    c=np.mean(Eref.real,0)
    d=np.mean(Eref.imag,0)
    
    k=np.mean(Esamcorrf.real,0)
    l=np.mean(Esamcorrf.imag,0)
    m=np.mean(Erefcorrf.real,0)
    n=np.mean(Erefcorrf.imag,0)

    anoise=a[noiseposition::]
    bnoise=b[noiseposition::]
    cnoise=c[noiseposition::]
    dnoise=d[noiseposition::]
    knoise=k[noiseposition::]
    lnoise=l[noiseposition::]
    mnoise=m[noiseposition::]
    nnoise=n[noiseposition::]
    
    anstd=np.std(anoise)
    bnstd=np.std(bnoise)
    cnstd=np.std(cnoise)
    dnstd=np.std(dnoise)
    knstd=np.std(knoise)
    lnstd=np.std(lnoise)
    mnstd=np.std(mnoise)
    nnstd=np.std(nnoise)

    A=Esample.real
    B=Esample.imag
    C=Eref.real
    D=Eref.imag
    K=Esamcorrf.real
    L=Esamcorrf.imag
    M=Erefcorrf.real
    N=Erefcorrf.imag

    Astd=np.std(A)
    Bstd=np.std(B)
    Cstd=np.std(C)
    Dstd=np.std(D)
    Kstd=np.std(K)
    Lstd=np.std(L)
    Mstd=np.std(M)
    Nstd=np.std(N)

    DeltaA=(Astd*Astd+anstd*anstd)**0.5
    DeltaB=(Bstd*Bstd+bnstd*bnstd)**0.5
    DeltaC=(Cstd*Cstd+cnstd*cnstd)**0.5
    DeltaD=(Dstd*Dstd+dnstd*dnstd)**0.5    
    DeltaK=(Kstd*Kstd+knstd*knstd)**0.5
    DeltaL=(Lstd*Lstd+lnstd*lnstd)**0.5
    DeltaM=(Mstd*Mstd+mnstd*mnstd)**0.5
    DeltaN=(Nstd*Nstd+nnstd*nnstd)**0.5

    
    dfda=(c*k*m + c*l*n + d*k*n - d*l*m)/((c*c + d*d)*(k*k + l*l))
    dfdb=(-c*k*n + c*l*m + d*k*m + d*l*n)/((c*c + d*d)*(k*k + l*l))
    dfdc=-(a*c*(c*k*m + c*l*n + 2*d*k*n - 2*d*l*m) + a*d*d*(-k*m - l*n) + b*c*(-c*k*n + c*l*m + 2*d*k*m + 2*d*l*n) + b*d*d*(k*n - l*m))/((c*c + d*d)**2 * (k*k + l*l))
    dfdd=(a*c*(c*k*n - c*l*m - 2*d*k*m - 2*d*l*n) + a*d*d*(-k*n + l*m) + b*c*(c*k*m + c*l*n + 2*d*k*n - 2*d*l*m) + b*d*d*(-k*m + -l*n))/((c*c + d*d)**2 * (k*k + l*l))
    dfdk=-(a*c*(k*k*m + 2*k*l*n - l*l*m) + a*d*(k*k*n - 2*k*l*m - l*l*n) + b*c*(-k*k*n + 2*k*l*m + l*l*n) + b*d*(k*k*m + 2*k*l*n - l*l*m))/((c*c + d*d)*(k*k + l*l)**2)
    dfdl=(a*c*(k*k*n - 2*k*l*m - l*l*n) + a*d*(-k*k*m - 2*k*l*n + l*l*m) + b*c*(k*k*m + 2*k*l*n - l*l*m) + b*d*(k*k*n - 2*k*l*m - l*l*n))/((c*c + d*d)*(k*k + l*l)**2)
    dfdm=(a*c*k - a*d*l + b*c*l + b*d*k)/((c*c + d*d)*(k*k + l*l))
    dfdn=(a*c*l + a*d*k - b*c*k + b*d*l)/((c*c + d*d)*(k*k + l*l))

    dgda=(c*k*n - c*l*m - d*k*m - d*l*n)/((c*c + d*d)*(k*k + l*l))
    dgdb=(c*k*m - d*l*m + d*k*n + c*l*n)/((c*c + d*d)*(k*k + l*l))
    dgdc=-(a*c*(c*k*n - c*l*m - 2*d*k*m - 2*d*l*n) + a*d*d*(-k*n + l*m) + b*c*(c*k*m + c*l*n + 2*d*k*n - 2*d*l*m) + b*d*d*(-k*m - l*n))/((c*c + d*d)**2 * (k*k + l*l))
    dgdd=(-a*c*(c*k*m - c*l*n - 2*d*k*n + 2*d*l*m) + a*d*d*(k*m + l*n) + b*c*(c*k*n - c*l*m - 2*d*k*m - 2*d*l*n) + b*d*d*(-k*n + l*m))/((c*c + d*d)**2 * (k*k + l*l))
    dgdk=-(a*c*(k*k*n - 2*k*l*m - l*l*n) + a*d*(-k*k*m - 2*k*l*n + l*l*m) + b*c*(k*k*m + 2*k*l*n - l*l*m) + b*d*(k*k*n - 2*k*l*m - l*l*n))/((c*c + d*d)*(k*k + l*l)**2)
    dgdl=-(a*c*(k*k*m + 2*k*l*n - l*l*m) + a*d*(k*k*n - 2*k*l*m - l*l*n) + b*c*(-k*k*n + 2*k*l*m + l*l*n) + b*d*(k*k*m + 2*k*l*n - l*l*m))/((c*c + d*d)*(k*k + l*l)**2)
    dgdm=-(a*c*l + a*d*k - b*c*k + b*d*l)/((c*c + d*d)*(k*k + l*l))
    dgdn=(a*c*k - a*d*l + b*c*l + b*d*k)/((c*c + d*d)*(k*k + l*l))

    DeltaRealH=((DeltaA*dfda)**2+(DeltaB*dfdb)**2+(DeltaC*dfdc)**2+(DeltaD*dfdd)**2+(DeltaK*dfdk)**2+(DeltaL*dfdl)**2+(DeltaM*dfdm)**2+(DeltaN*dfdn)**2)**0.5
    DeltaImagH=((DeltaA*dgda)**2+(DeltaB*dgdb)**2+(DeltaC*dgdc)**2+(DeltaD*dgdd)**2+(DeltaK*dgdk)**2+(DeltaL*dgdl)**2+(DeltaM*dgdm)**2+(DeltaN*dgdn)**2)**0.5

    return DeltaRealH+1j*DeltaImagH

#errH=TransferfunctionError(FDdata2['sample1'],FDdata2['ref'],FDdata2['sample1_c'],FDdata2['ref_c'],noiseposition(f,1.4))

def TransferfunctionError2(Esample,Eref,Esamcorrf,Erefcorrf,noiseposition):#Since Correctionfunction gets only used as mean in Transferfunction
    #H=Esam/Eref*1/Fcor
    #Fcor=mean(Esamcorr/Erefcorr)
    #H=Esam*Erefcorr/mean(Eref*Esamcorr)=((a+ib)*(m+in))/mean((c+id)*(k+il))
    #f=Re(H)
    #g=Im(H)
    a=np.mean(Esample.real,0)
    b=np.mean(Esample.imag,0)
    c=np.mean(Eref.real,0)
    d=np.mean(Eref.imag,0)
    
    k=np.mean(Esamcorrf.real,0)
    l=np.mean(Esamcorrf.imag,0)
    m=np.mean(Erefcorrf.real,0)
    n=np.mean(Erefcorrf.imag,0)

    anoise=a[noiseposition::]
    bnoise=b[noiseposition::]
    cnoise=c[noiseposition::]
    dnoise=d[noiseposition::]
    knoise=k
    lnoise=l
    mnoise=m
    nnoise=n
    
    anstd=np.std(anoise)
    bnstd=np.std(bnoise)
    cnstd=np.std(cnoise)
    dnstd=np.std(dnoise)
    knstd=np.std(knoise)
    lnstd=np.std(lnoise)
    mnstd=np.std(mnoise)
    nnstd=np.std(nnoise)

    A=Esample.real
    B=Esample.imag
    C=Eref.real
    D=Eref.imag

    Astd=np.std(A)
    Bstd=np.std(B)
    Cstd=np.std(C)
    Dstd=np.std(D)

    DeltaA=(Astd*Astd+anstd*anstd)**0.5
    DeltaB=(Bstd*Bstd+bnstd*bnstd)**0.5
    DeltaC=(Cstd*Cstd+cnstd*cnstd)**0.5
    DeltaD=(Dstd*Dstd+dnstd*dnstd)**0.5    
    DeltaK=knstd
    DeltaL=lnstd
    DeltaM=mnstd
    DeltaN=nnstd
    
    dfda=(c*k*m + c*l*n + d*k*n - d*l*m)/((c*c + d*d)*(k*k + l*l))
    dfdb=(-c*k*n + c*l*m + d*k*m + d*l*n)/((c*c + d*d)*(k*k + l*l))
    dfdc=-(a*c*(c*k*m + c*l*n + 2*d*k*n - 2*d*l*m) + a*d*d*(-k*m - l*n) + b*c*(-c*k*n + c*l*m + 2*d*k*m + 2*d*l*n) + b*d*d*(k*n - l*m))/((c*c + d*d)**2 * (k*k + l*l))
    dfdd=(a*c*(c*k*n - c*l*m - 2*d*k*m - 2*d*l*n) + a*d*d*(-k*n + l*m) + b*c*(c*k*m + c*l*n + 2*d*k*n - 2*d*l*m) + b*d*d*(-k*m + -l*n))/((c*c + d*d)**2 * (k*k + l*l))
    dfdk=-(a*c*(k*k*m + 2*k*l*n - l*l*m) + a*d*(k*k*n - 2*k*l*m - l*l*n) + b*c*(-k*k*n + 2*k*l*m + l*l*n) + b*d*(k*k*m + 2*k*l*n - l*l*m))/((c*c + d*d)*(k*k + l*l)**2)
    dfdl=(a*c*(k*k*n - 2*k*l*m - l*l*n) + a*d*(-k*k*m - 2*k*l*n + l*l*m) + b*c*(k*k*m + 2*k*l*n - l*l*m) + b*d*(k*k*n - 2*k*l*m - l*l*n))/((c*c + d*d)*(k*k + l*l)**2)
    dfdm=(a*c*k - a*d*l + b*c*l + b*d*k)/((c*c + d*d)*(k*k + l*l))
    dfdn=(a*c*l + a*d*k - b*c*k + b*d*l)/((c*c + d*d)*(k*k + l*l))

    dgda=(c*k*n - c*l*m - d*k*m - d*l*n)/((c*c + d*d)*(k*k + l*l))
    dgdb=(c*k*m - d*l*m + d*k*n + c*l*n)/((c*c + d*d)*(k*k + l*l))
    dgdc=-(a*c*(c*k*n - c*l*m - 2*d*k*m - 2*d*l*n) + a*d*d*(-k*n + l*m) + b*c*(c*k*m + c*l*n + 2*d*k*n - 2*d*l*m) + b*d*d*(-k*m - l*n))/((c*c + d*d)**2 * (k*k + l*l))
    dgdd=(-a*c*(c*k*m - c*l*n - 2*d*k*n + 2*d*l*m) + a*d*d*(k*m + l*n) + b*c*(c*k*n - c*l*m - 2*d*k*m - 2*d*l*n) + b*d*d*(-k*n + l*m))/((c*c + d*d)**2 * (k*k + l*l))
    dgdk=-(a*c*(k*k*n - 2*k*l*m - l*l*n) + a*d*(-k*k*m - 2*k*l*n + l*l*m) + b*c*(k*k*m + 2*k*l*n - l*l*m) + b*d*(k*k*n - 2*k*l*m - l*l*n))/((c*c + d*d)*(k*k + l*l)**2)
    dgdl=-(a*c*(k*k*m + 2*k*l*n - l*l*m) + a*d*(k*k*n - 2*k*l*m - l*l*n) + b*c*(-k*k*n + 2*k*l*m + l*l*n) + b*d*(k*k*m + 2*k*l*n - l*l*m))/((c*c + d*d)*(k*k + l*l)**2)
    dgdm=-(a*c*l + a*d*k - b*c*k + b*d*l)/((c*c + d*d)*(k*k + l*l))
    dgdn=(a*c*k - a*d*l + b*c*l + b*d*k)/((c*c + d*d)*(k*k + l*l))

    DeltaRealH=((DeltaA*dfda)**2+(DeltaB*dfdb)**2+(DeltaC*dfdc)**2+(DeltaD*dfdd)**2+(DeltaK*dfdk)**2+(DeltaL*dfdl)**2+(DeltaM*dfdm)**2+(DeltaN*dfdn)**2)**0.5
    DeltaImagH=((DeltaA*dgda)**2+(DeltaB*dgdb)**2+(DeltaC*dgdc)**2+(DeltaD*dgdd)**2+(DeltaK*dgdk)**2+(DeltaL*dgdl)**2+(DeltaM*dgdm)**2+(DeltaN*dgdn)**2)**0.5

    return DeltaRealH+1j*DeltaImagH


def ReflectionfactorError(n1,deltan1,n2,deltan2,Theta1,deltaTheta1): 
    cosTheta2=np.sqrt(np.complex128(1-(n1*n1)/(n2*n2)*np.sin(Theta1)**2))
    dct2dn1=-(n1*np.sin(Theta1)**2)/(n2*n2*(1 - (n1*n1*np.sin(Theta1)**2)/(n2*n2))**0.5)
    dct2dn2=(n1*n1*np.sin(Theta1)**2)/(n2*n2*n2*(1 - (n1*n1*np.sin(Theta1)**2)/(n2*n2))**0.5)
    dct2dt1=-(n1*n1*np.sin(Theta1)*np.cos(Theta1))/(n2*n2*(1 - (n1*n1*np.sin(Theta1)**2)/(n2*n2))**0.5)
    
    deltaCosTheta2Real=((deltan1.real*dct2dn1.real)**2+(deltan2.real*dct2dn2.real)**2+(deltaTheta1.real*dct2dt1.real)**2)**0.5
    deltaCosTheta2Imag=((deltan1.imag*dct2dn1.imag)**2+(deltan2.imag*dct2dn2.imag)**2+(deltaTheta1.imag*dct2dt1.imag)**2)**0.5
    
    #rp12=(n2*np.cos(Theta1)-n1*cos_th2)/(n2*np.cos(Theta1)+n1*cos_th2)
    drdn1=-(2*cosTheta2*n2*np.cos(Theta1))/(cosTheta2*n1 + n2*np.cos(Theta1))**2
    drdn2=(2*cosTheta2*n1*np.cos(Theta1))/(cosTheta2*n1 + n2*np.cos(Theta1))**2
    drdt1=-(2*cosTheta2*n1*n2*np.sin(Theta1))/(cosTheta2*n1 + n2*np.cos(Theta1))**2
    drdct2=-(2*n1*n2*np.cos(Theta1))/(cosTheta2*n1 + n2*np.cos(Theta1))**2

    deltarReal=((deltan1.real*drdn1.real)**2+(deltan2.real*drdn2.real)**2+(deltaTheta1.real*drdt1.real)**2+(deltaCosTheta2Real*drdct2.real)**2)**0.5
    deltarImag=((deltan1.imag*drdn1.imag)**2+(deltan2.imag*drdn2.imag)**2+(deltaTheta1.imag*drdt1.imag)**2+(deltaCosTheta2Imag*drdct2.imag)**2)**0.5
    
    return deltarReal+1j*deltarImag


def RefractiveIndexError(n1,deltan1,Theta1,deltaTheta1,TF,deltaTF):
    def r_p(n1,n2,Theta1):
        cos_th2 = np.sqrt(np.complex128(1-((n1/n2)**2)*np.sin(Theta1)**2))
        r_coeff=(n2*np.cos(Theta1)-n1*cos_th2)/(n2*np.cos(Theta1)+n1*cos_th2)
        return r_coeff
    
    r_cal=r_p(n1,1,Theta1)
    deltar_cal=ReflectionfactorError(n1,deltan1,1+0j,0+0j,Theta1,deltaTheta1)
    
    rp=r_cal*TF
    
    deltarpReal=((TF.real*deltar_cal.real)**2+(r_cal.real*deltaTF.real)**2)**0.5
    deltarpImag=((TF.imag*deltar_cal.imag)**2+(r_cal.imag*deltaTF.imag)**2)**0.5
    
    
    q=(rp-1)/(rp+1)
    dqdrp=2.0/(rp+1)**2
    
    deltaqReal=dqdrp.real*deltarpReal
    deltaqImag=dqdrp.imag*deltarpImag
    
    
    Delta=1-q**2*(np.sin(2*Theta1))**2
    dDeltadq=-2*q*(np.sin(2*Theta1))**2
    dDeltadTheta1=-q*q*4*np.sin(2*Theta1)*np.cos(2*Theta1)
    
    deltaDeltaReal=((deltaqReal*dDeltadq.real)**2+(deltaTheta1.real*dDeltadTheta1.real)**2)**0.5
    deltaDeltaImag=((deltaqImag*dDeltadq.imag)**2+(deltaTheta1.imag*dDeltadTheta1.imag)**2)**0.5
    
    
    eps=(n1**2)*((1-np.sqrt(Delta))/(2*q**2*(np.cos(Theta1))**2))
    
    deps1dn1=2*n1*((1-np.sqrt(Delta))/(2*q*q*(np.cos(Theta1))**2))
    deps1dq=-(n1**2)*((1-np.sqrt(Delta))/(q*q*q*(np.cos(Theta1))**2))
    deps1dDelta=-(n1**2)/(4*np.sqrt(Delta)*q*q*(np.cos(Theta1))**2)
    deps1dTheta1=(n1**2)*((1-np.sqrt(Delta))*np.tan(Theta1)/(q*q*(np.cos(Theta1))**2))
    
    deltaeps1Real=((deltan1.real*deps1dn1.real)**2+(deltaqReal*deps1dq.real)**2+(deltaDeltaReal*deps1dDelta.real)**2+(deltaTheta1.real*deps1dTheta1.real)**2)**0.5
    deltaeps1Imag=((deltan1.imag*deps1dn1.imag)**2+(deltaqImag*deps1dq.imag)**2+(deltaDeltaImag*deps1dDelta.imag)**2+(deltaTheta1.imag*deps1dTheta1.imag)**2)**0.5
    
    deps2dn1=2*n1*((1+np.sqrt(Delta))/(2*q*q*(np.cos(Theta1))**2))
    deps2dq=-(n1**2)*((1+np.sqrt(Delta))/(q*q*q*(np.cos(Theta1))**2))
    deps2dDelta=(n1**2)/(4*np.sqrt(Delta)*q*q*(np.cos(Theta1))**2)
    deps2dTheta1=(n1**2)*((1+np.sqrt(Delta))*np.tan(Theta1)/(q*q*(np.cos(Theta1))**2))
    
    deltaeps2Real=((deltan1.real*deps2dn1.real)**2+(deltaqReal*deps2dq.real)**2+(deltaDeltaReal*deps2dDelta.real)**2+(deltaTheta1.real*deps2dTheta1.real)**2)**0.5
    deltaeps2Imag=((deltan1.imag*deps2dn1.imag)**2+(deltaqImag*deps2dq.imag)**2+(deltaDeltaImag*deps2dDelta.imag)**2+(deltaTheta1.imag*deps2dTheta1.imag)**2)**0.5
    
    
    deltaepsReal=deltaeps1Real
    deltaepsImag=deltaeps1Imag

    idx = np.imag(eps) < 0
    if(type(idx)==np.bool_):
        if(idx):
            eps=(n1**2)*((1+np.sqrt(Delta))/(2*q**2*(np.cos(Theta1))**2))
            deltaepsReal=deltaeps2Real
            deltaepsImag=deltaeps2Imag
    else:
        eps[idx] = (n1**2)*((1+np.sqrt(Delta))/(2*q**2*(np.cos(Theta1))**2))[idx]
        deltaepsReal[idx]=deltaeps2Real[idx]
        deltaepsImag[idx]=deltaeps2Imag[idx]
    
    #n = np.sqrt(eps)
    dndeps=0.5/np.sqrt(eps)
    
    deltanReal=dndeps.real*deltaepsReal
    deltanImag=dndeps.imag*deltaepsImag
    
    return deltanReal+1j*deltanImag







#since average is used: using errorpropagation of meanformula for errorbars (mean=sum x_i/num, s=sqrt(sum (dmean/dxi*deltaxi)^2)) --> dmean/dxi=1/num --> s=1/num*sqrt(sum deltaxi^2)







