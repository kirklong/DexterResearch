#!/usr/bin/env python
import numpy, pickle, emcee
from multiprocessing import Pool, cpu_count
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d

def readPickle(file):
    with open(file,"rb") as f:
        data = pickle.load(f,encoding="latin1")
    return data

def dvldl(r,sini,cosi,φ,windWeight=0,f1=1,f2=1,f3=1): #line of sight velocity gradient
    windφ1 = 3*numpy.sqrt(1/(2*r))/r*sini**2*numpy.cos(φ)*(numpy.sqrt(2)*numpy.cos(φ)+numpy.sin(φ)/2)
    windφ2 = cosi**2*(1/(r**(3/2)))
    windφ3 = -3*numpy.sqrt(1/(2*r))/r*sini*cosi
    diskφ = 3*numpy.sqrt(1/(2*r))/r*sini**2*numpy.cos(φ)*(numpy.cos(φ)*numpy.sin(φ)/2)
    dvl = (1-windWeight)*diskφ+windWeight*(f1*windφ1+f2*windφ2+f3*windφ3)
    return dvl

def getA(A0,x,γ):
    return A0*x**γ

def intensity(A,r,grad_v,τ):
    return A/(4*numpy.pi*r**2)*numpy.abs(grad_v)*(1-numpy.exp(-τ))

def setup(i=75,nx=2048,ny=2048,rlim=3e4):
    a = numpy.arange(nx)/(nx-1)*rlim-rlim/2; b = numpy.arange(ny)/(ny-1)*rlim-rlim/2
    α,β = numpy.meshgrid(a,b)
    i = i/180*numpy.pi; cosi = numpy.cos(i); sini = numpy.sin(i)

    r = numpy.sqrt(β**2/cosi**2+α**2); φ = numpy.arctan2(β,α*cosi)
    ν = 1+numpy.sqrt(1/(2*r))*sini*numpy.cos(φ)
    return α,β,r,ν,φ,sini,cosi
iter
def getIntensity(r,φ,windWeight,sini,cosi,rMin=1e3,γ=1,A0=1,τ=10,f1=1,f2=1,f3=1):
    φn = φ + numpy.pi/2
    grad_v = dvldl(r,sini,cosi,φn,windWeight,f1,f2,f3)
    A = getA(A0,r,γ)
    I = intensity(A,r,grad_v,τ); I[r<rMin] = 0
    return I,γ,A0,τ

def phase(ν,I,x,y,U,V,bins=100):
    dφMap = -2*numpy.pi*(x*U+y*V)*I*180/numpy.pi*1e6
    dφ,edges,n = binned_statistic(ν.flatten(),dφMap.flatten(),statistic="sum",bins=bins)
    iSum,edges,n = binned_statistic(ν.flatten(),I.flatten(),statistic="sum",bins=bins)
    iSum[iSum==0]=1
    return dφ/iSum

def getProfiles(ν,params,data,bins=100,nx=2048,ny=2048):
    i,rMin,Mfac,rFac,f1,f2,f3=params; windWeight=1 #setting to 1 perm for now
    blRange=Mfac*3e8*2e33*6.67e-8/9e20/548/3.09e24
    α,β,r,νloc,φ,sini,cosi = setup(i,nx,ny)
    I,γ,A0,τ = getIntensity(r,φ,windWeight,sini,cosi,rMin,f1=f1,f2=f2,f3=f3)
    flux,νEdges,n = binned_statistic(νloc.flatten(),I.flatten(),statistic="sum",bins=100)
    νBin = 0.5*(νEdges[1:]+νEdges[:-1])
    UData=data[1];VData=data[2]; psf=4e-3/2.35
    X=α*blRange; Y=β*blRange
    dφList=[]
    for i in range(len(UData)):
        for ii in [I]:
            dφAvgRaw=phase(νloc,ii,X,Y,UData[i],VData[i],bins)
            dφAvg=gaussian_filter1d(dφAvgRaw,psf/3e5/(νBin[1]-νBin[0]))
            dφList.append(dφAvg)
    fline=flux/numpy.max(flux)*0.6/(1+flux/numpy.max(flux)*0.6)
    indx=[0,1,2,6,7,8,12,13,14,18,19,20]; oindx=[3,4,5,9,10,11,15,16,17,21,22,23]
    x=(νBin-1)*3e5; yP=numpy.mean(numpy.array(dφList)[indx],axis=0)*fline*rFac; yPo=numpy.mean(numpy.array(dφList)[oindx],axis=0)*fline*rFac
    interpPhase = numpy.interp(ν,x,yP); interpPhaseo = numpy.interp(ν,x,yPo)
    yL = flux/numpy.max(flux)*numpy.max(data[3])
    interpLine = numpy.interp(ν,x,yL)
    return interpLine,interpPhase,interpPhaseo

def log_lhood(θ,x,data):
    lineInterpY,phaseInterpY,phaseoInterpY = getProfiles(x,θ,data)
    yLErr = data[7]; yPErr = x*0.0+0.07 #phase error wrong, should fit each individually
    indx=[0,1,2,6,7,8,12,13,14,18,19,20]; oindx=[3,4,5,9,10,11,15,16,17,21,22,23]
    lnLikeLine = -0.5*numpy.sum(((data[3]-lineInterpY)/yLErr)**2)
    lnLikePhase = -0.5*numpy.sum(((numpy.mean(numpy.array(data[4])[indx],axis=0) - phaseInterpY)/yPErr)**2)
    lnLikePhase = -0.5*numpy.sum(((numpy.mean(numpy.array(data[4])[oindx],axis=0) - phaseInterpY)/yPErr)**2)
    return lnLikeLine + lnLikePhase + lnLikePhaseo

def log_prior(θ):
    i,rMin,Mfac,rFac,f1,f2,f3 = θ
    if i>0 and i<90 and rMin>500 and rMin<1e4 and Mfac>0 and rFac>0 and f1>=1 and f1<=1 and f2>=0 and f2<=1 and f3>=0 and f3<=1:
        return 0.0
    else:
        return -numpy.Inf

def log_prob(θ,x,data):
    lnP = log_prior(θ)
    if lnP == -numpy.Inf:
        return -numpy.Inf
    else:
        return lnP + log_lhood(θ,x,data)

def MC(nWalkers,θ0,p0,log_prob,vel,data,threads,burn=100,iter=1000):
    with Pool(threads) as pool:
        sampler = emcee.EnsembleSampler(nWalkers,len(θ0),log_prob,args=(vel,data),pool=pool)
        print("running burn-in")
        p0,_,_ = sampler.run_mcmc(p0,burn,skip_initial_state_check=True,progress=True)
        sampler.reset()
        print("production run (n = {0} iterations)".format(iter))
        pos,prob,state = sampler.run_mcmc(p0,iter,skip_initial_state_check=True,progress=True)
        return sampler,pos,prob,state

def main(specifyThreads = False):
    if specifyThreads == False:
        threads = cpu_count() #for Summit job use all of them
    else:
        threads = 4 #for my computer
    print("running with {} threads".format(threads))
    data = readPickle("3c273_juljanmarmay_append_gilles_specirf_wide_v6.p")
    λCen = 2.172
    vel = (data[0]-λCen)/λCen*3e5
    pert = [45.,100.,0.5,0.5,0.5,0.5,0.5]
    nWalkers = 50; θ0 = [45.,1e3,1.,1.3,0.5,0.5,0.5]
    p0 = numpy.zeros((len(θ0),nWalkers))
    for n in range(len(pert)):
        p0[n] = [θ0[n]+pert[n]*numpy.random.randn(1) for j in range(nWalkers)]
    p0 = numpy.transpose(p0)
    sampler,pos,prob,state = MC(nWalkers,θ0,p0,log_prob,vel,data,threads)
    flat_samples = sampler.get_chain(flat=True)
    return sampler,pos,prob,state,flat_samples
