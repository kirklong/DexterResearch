#!/usr/bin/env python
import julia, pickle, numpy
import ptemcee
import multiprocessing
multiprocessing.set_start_method("spawn",force=True) #required for julia compat
Pool = multiprocessing.Pool; cpu_count = multiprocessing.cpu_count
#reference for figuring this out: https://stackoverflow.com/questions/64241264/i-have-a-high-performant-function-written-in-julia-how-can-i-use-it-from-python

#try putting block here
jl = julia.Julia(compiled_modules=False)
from julia import Pkg
Pkg.activate("DiskWind")
from julia import DiskWind


def readPickle(file):
    with open(file,"rb") as f:
        data = pickle.load(f,encoding="latin1")
    return data

def log_lhood(θ,data):
    ν,lineInterpY,phaseInterpList = DiskWind.getProfiles(θ,data)
    yLErr = data[6]; #yPErr = x*0.0+0.07 #phase error wrong, should fit each individually
    indx=[0,1,2,6,7,8,12,13,14,18,19,20]; oindx=[3,4,5,9,10,11,15,16,17,21,22,23]
    lnLikeLine = -0.5*numpy.sum(((data[3]-lineInterpY)/yLErr)**2)
    lnLikePhase = numpy.sum([-0.5*numpy.sum(((data[4][i] - phaseInterpList[i])/data[5][i,:])**2) for i in range(len(phaseInterpList))]) #so it's weighted equally, UPDATE don't do, because no differnce in χ2 when comparing avg vs individual profiles
    return lnLikeLine + lnLikePhase #+ lnLikePhaseo

def log_prior(θ):
    i,rBar,Mfac,rFac,f1,f2,f3,f4,pa,scale,cenShift = θ
    if i>0 and i<90 and rBar>250 and rBar<1e4 and Mfac>0 and rFac>1 and f1>=0 and f1<=1 and f2>=0 and f2<=1 and f3>=0 and f3<=1 and f4>=0 and f4<=1 and pa>=0 and pa<360 and scale>=1 and numpy.abs(cenShift) < 0.1:
        return 0.0
    else:
        return -numpy.Inf

def log_prob(θ,data):
    lnP = log_prior(θ)
    if lnP == -numpy.Inf:
        return -numpy.Inf
    else:
        return lnP + log_lhood(θ,data)

def MC(nWalkers,nTemps,θ0,p0,log_prob,data,threads,burn=100,iter=100,restart=False):
    print("initializing sampler")
    #sampler = emcee.EnsembleSampler(nWalkers,len(θ0),log_prob,args=[data],pool=pool)
    sampler = ptemcee.sampler.Sampler(nWalkers,len(θ0),log_lhood,log_prior,ntemps=nTemps,loglargs=[data],threads=threads)
    if restart == False:
        print("running burn-in")
        p0,_,_ = sampler.run_mcmc(p0,burn)#,skip_initial_state_check=True,progress=True)
        sampler.reset()
        print("initial production run (n = {0} iterations)".format(iter))
    else:
        print("reading in last saved position")
        flat_samples,pos,prob,lhood,acor = readPickle("jPyPTEmceeVar.p")
        print("resuming -- running for {0} more iterations".format(iter))
        p0 = pos
    pos,prob,state = sampler.run_mcmc(p0,iter)#,skip_initial_state_check=True,progress=True)
    return sampler,pos,prob,state

def main(specifyThreads = False, save = True, burn=100,iter=3000, restart=False):
    if specifyThreads == False:
        threads = cpu_count() #for Summit job use all of them
    else:
        threads = 4 #for my computer

    print("running with {} threads".format(threads))
    data = readPickle("3c273_juljanmarmay_append_gilles_specirf_wide_v6.p")
    pert = numpy.array([40.,1e3,0.5,15.,0.4,0.4,0.4,0.4,50.,0.01,0.001])*0.1 #i,rBar,Mfac,rFac,f1,f2,f3, pa, scale, cenShift, try for 0.5% around initial guess
    nWalkers = 24; nTemps=6; θ0 = numpy.array([45.,3e3,1.,30.,0.5,0.5,0.5,0.5,300,1.01,0.])#[30.,1e3,1.1,1.,0.57,0.6,0.46,342.] #i,rBar,Mfac,rFac,f1,f2,f3,f4,pa,scale,cenShift
    p0 = numpy.zeros((nTemps,nWalkers,len(θ0)))
    for n in range(len(pert)):
        p0[:,:,n] = numpy.array([θ0[n]+pert[n]*numpy.random.randn(1) for j in range(nWalkers*nTemps)]).reshape(nTemps,nWalkers)
    sampler,pos,prob,state = MC(nWalkers,nTemps,θ0,p0,log_prob,data,threads,burn,iter,restart)
    flat_samples = sampler.flatchain#(flat=True)
    prob = sampler.logprobability #this is the prob for everyone
    lhood = sampler.loglikelihood
    prob = prob.reshape(*prob.shape[:1],-1) #collapse along walker axis so it's same shape as flat_chain
    lhood = prob.reshape(*prob.shape[:1],-1) #collapse along walker axis so it's same shape as flat_chain
    acor = sampler.acor #autocorrelation
    if save == True: #ie on summitsaml
        with open("jPyPTEmceeVar.p","wb") as f:
            obj = [flat_samples,pos,prob,lhood,acor] #really only care about flat_samples, don't think we need sampler/state
            pickle.dump(obj,f)
    else: #ie for interactive nb
        return sampler,pos,prob,state,flat_samples

#main(True)
