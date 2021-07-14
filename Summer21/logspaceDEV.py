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

def setup(i=30.,nρ=1024,nΦ=1024,rMin=1e3,rLim=1.5e4):
    #a = numpy.arange(nx)/(nx-1)*rlim-rlim/2; b = numpy.arange(ny)/(ny-1)*rlim-rlim/2
    def ellipse(r,θ,cosi):
        A = r; B = r*cosi #ellipse is shrunk in y direction by factor of cosi projection
        e = numpy.sqrt(1-(B/A)**2)
        return B/numpy.sqrt(1-(e*numpy.cos(θ))**2)

    i = i/180*numpy.pi; cosi = numpy.cos(i); sini = numpy.sin(i)
    r = numpy.logspace(numpy.log(rMin*cosi),numpy.log(rLim),num=nρ,base=numpy.e);
    #r = numpy.linspace(rMin*cosi,rLim,nρ) #try doing it not in log space
    Δr = r[1]-r[0]
    φ = numpy.linspace(0,2*numpy.pi,nΦ,endpoint=False) #r and φ initially (camera), don't include 2π because degenerate with 0
    Δlnr = numpy.log(r[1]/r[0]); Δφ = φ[1]-φ[0]
    Δr = [r[i]*(numpy.exp(Δlnr)-1) if i>0 else 1.0 for i in range(nρ)]
    #NEW/LAZY WAY
    rMesh,φMesh = numpy.meshgrid(r,φ)
    α,β = rMesh*numpy.cos(φMesh), rMesh*numpy.sin(φMesh)#now make it cartesian
    #x,y = a*np.cos(b), a*np.sin(b)
    #β = numpy.array([φCam for r in rCam]) #make it 2d, ie poor man's meshgrid
    #α = numpy.array([ellipse(r,φCam,cosi) for r in rCam]) #apply ellipse rules to 2d r array
    # rCam,φCam = numpy.zeros((nΦ,nρ)), numpy.zeros((nΦ,nρ)) #keep forgetting python is row major
    # r4int = numpy.array([r for i in range(nΦ)]) #because of how we set up dA formula only want the "a" of the ellipse, not b
    # for i in range(nρ):
    #     rCam[:,i] = ellipse(r[i],φ,cosi) #ellipse runs "down" the array (φ direction) while r goes from min to max across
    #     φCam[:,i] = φ #0 to 2π runs "down" the array, repeats for every r value across
    # #α,β = numpy.meshgrid(rCam,φCam)

    # #################### CHRISTIAN'S IDEA ##########################################
    # dA = numpy.zeros((nΦ,nρ)) #initalize array
    # α = r; β = r*cosi #Christian's α and β, r is my original list of "a" values (included below for completeness, commented out because it's initialized elsewhere)
    # # below are how r, φ are initialized (used φ instead of θ)
    # # r = numpy.logspace(numpy.log(rMin),numpy.log(rLim),num=nρ,base=numpy.e);
    # # φ = numpy.linspace(0,2*numpy.pi,nΦ,endpoint=False) #r and φ initially (camera), don't include 2π because degenerate with 0
    # Δθ = φ[1]-φ[0] #stays constant across ellipse, was originally using φ variable for this, see above
    # for i in range(nΦ-1): #i index corresponds to θ/φ space
    #     for j in range(nρ-1): #j index corresponds to increasing r/a/b
    #         term1 = α[j]*β[j]/((β[j]*numpy.cos(φ[i]))**2+(α[j]*numpy.sin(φ[i]))**2)
    #         term2 = α[j+1]*β[j+1]/((β[j+1]*numpy.cos(φ[i]))**2+(α[j+1]*numpy.sin(φ[i]))**2)
    #         dA[i,j] = Δθ*(numpy.sqrt(term1*term2)-term1) #pretty sure this is your ending formula, but you should make sure you agree
    # ##################################################################################

    #α,β = rCam*numpy.cos(φCam), rCam*numpy.sin(φCam)#now make it cartesian
    r = numpy.sqrt(β**2/cosi**2+α**2); φ = numpy.arctan2(β,α*cosi) #physical r and φ, raytraced backwards from elliptical camera "image"
    ν = 1+numpy.sqrt(1/(2*r))*sini*numpy.cos(φ)
    #φPart = numpy.arctan(1/cosi*numpy.tan(φCam+Δφ))-numpy.arctan(1/cosi*numpy.tan(φCam-Δφ)); φPart[φPart<0]=numpy.max(φPart) #near +/- π/2 it is not numerically stable, but this is what analytic behavior should be (only applies to like 6 cells, should probably calculate indices...)
    #dA = r4int**2*numpy.sinh(Δlnr)*cosi*φPart #want area elements on CAMERA (only sum up intensity of image, not physical)
    dA = numpy.zeros((nΦ,nρ)); #Δr = numpy.zeros(nρ)
    # #Δr[1:] = [rMesh[0,i+1]-rMesh[0,i] for i in range(nρ-1)]
    for i in range(nΦ):
        dA[i,:] = rMesh[i,:]*Δr*Δφ
    #dA = rMesh**2*numpy.exp(Δlnr)*Δφ #this is camera dA
    #dA = rMesh*Δr*Δφ
    return α,β,r,ν,φ,sini,cosi,dA

def getIntensity(r,φ,windWeight,sini,cosi,rMin=1e3,γ=1,A0=1,τ=10,f1=1,f2=1,f3=1):
    φn = φ + numpy.pi/2
    grad_v = dvldl(r,sini,cosi,φn,windWeight,f1,f2,f3)
    A = getA(A0,r,γ)
    I = intensity(A,r,grad_v,τ); I[r<=rMin] = 0
    return I,γ,A0,τ

def phase(ν,I,dA,x,y,r,U,V,rot,bins=100): #rot go between 0 and 2π, fit for this also (proxy for pos angle, need to do math to backtrack after fit though)
    rot = rot/180*numpy.pi
    un = numpy.cos(rot)*U+numpy.sin(rot)*V; vn = -numpy.sin(rot)*U+numpy.cos(rot)*V
    dφMap = -2*numpy.pi*(x*un+y*vn)*I*180/numpy.pi*1e6
    #Δlnr = numpy.abs(numpy.log(r[1][0])-numpy.log(r[0][0])); Δφ = numpy.abs(φ[0][1]-φ[0][0])
    dφ,edges,n = binned_statistic(ν.flatten(),(dφMap*dA).flatten(),statistic="sum",bins=bins)
    iSum,edges,n = binned_statistic(ν.flatten(),(I*dA).flatten(),statistic="sum",bins=bins)
    iSum[iSum==0]=1 #it's never zero anymore so save computation time
    return dφ/iSum

def getProfiles(ν,params,data,bins=100,nρ=1024,nΦ=1024): #get the phase and line profiles, matched to data
    i,rMin,Mfac,rFac,f1,f2,f3,pa=params; windWeight=1 #setting to 1 perm for now
    blRange=Mfac*3e8*2e33*6.67e-8/9e20/548/3.09e24  #solar masses * g / c^2 / Mpc -> end units = rad
    α,β,r,νloc,φ,sini,cosi,dA = setup(i,nρ,nΦ) #coordinate system / param setup
    I,γ,A0,τ = getIntensity(r,φ,windWeight,sini,cosi,rMin,f1=f1,f2=f2,f3=f3) #get the intensity from setup params
    Δlnr = numpy.log(r[0,1]/r[0,0]); Δφ = numpy.abs(φ[1,0]-φ[0,0])
    #flux,νEdges,n = binned_statistic(νloc.flatten(),(I*r**2*Δφ*numpy.sinh(Δlnr)).flatten(),statistic="sum",bins=100) #bin intensity with frequency
    flux,νEdges,n = binned_statistic(νloc.flatten(),(I*dA).flatten(),statistic="sum",bins=100)
    νBin = 0.5*(νEdges[1:]+νEdges[:-1])
    UData=data[1];VData=data[2]; psf=4e-3/2.35 #baseline data, psf value provided by Jason for Gaussian 1D filter
    X=α*blRange; Y=β*blRange #convert from angle to physical via blRange
    dφList=[]
    for i in range(len(UData)):
        for ii in [I]:
            dφAvgRaw=phase(νloc,ii,dA,X,Y,r,UData[i],VData[i],pa,bins) #get the binned phase
            dφAvg=gaussian_filter1d(dφAvgRaw,psf/3e5/(νBin[1]-νBin[0])) #apply filter
            dφList.append(dφAvg)
    fline=flux/numpy.max(flux)*0.6/(1+flux/numpy.max(flux)*0.6) #rescale flux

    fline = flux #temporary

    indx=[0,1,2,6,7,8,12,13,14,18,19,20]; oindx=[3,4,5,9,10,11,15,16,17,21,22,23]
    interpPhaseList = []; x=(νBin-1)*3e5; #x is ν bins in km/s
    for i in range(len(dφList)):
        yP=numpy.array(dφList[i])*fline*rFac #rescale phase by flux*rFac
        interpPhase = numpy.interp(ν,x,yP) #generate interpolation so we can compare directly to data points
        interpPhaseList.append(interpPhase)
    #yPo=numpy.mean(numpy.array(dφList)[oindx],axis=0)*fline*rFac
    # interpPhaseo = numpy.interp(ν,x,yPo)
    yL = flux/numpy.max(flux)*numpy.max(data[3]) #normalize flux

    yL = flux #temp

    interpLine = numpy.interp(ν,x,yL) #generate interpolation to directly compare to data
    return interpLine,interpPhaseList

def log_lhood(θ,x,data):
    lineInterpY,phaseInterpList = getProfiles(x,θ,data)
    yLErr = data[7]; #yPErr = x*0.0+0.07 #phase error wrong, should fit each individually
    indx=[0,1,2,6,7,8,12,13,14,18,19,20]; oindx=[3,4,5,9,10,11,15,16,17,21,22,23]
    lnLikeLine = -0.5*numpy.sum(((data[3]-lineInterpY)/yLErr)**2)
    #multiply = [1 if i in indx else 0 for i in range(24)] #24 different phase curves, want to zero out the "off" ones because they are messing up the fit, don't want to do a for loop -- jason said bad idea
    lnLikePhase = numpy.sum([-0.5*numpy.sum(((data[4][i] - phaseInterpList[i])/data[5][i,:])**2) for i in range(len(phaseInterpList))]) #so it's weighted equally, UPDATE don't do, because no differnce in χ2 when comparing avg vs individual profiles
    #lnLikePhase = -0.5*numpy.sum(((numpy.mean(numpy.array(data[4])[oindx],axis=0) - phaseInterpY)/yPErr)**2)
    return lnLikeLine + lnLikePhase #+ lnLikePhaseo

def log_prior(θ):
    i,rMin,Mfac,rFac,f1,f2,f3,pa = θ
    if i>0 and i<90 and rMin>250 and rMin<1e4 and Mfac>0 and rFac>0 and f1>=0 and f1<=1 and f2>=0 and f2<=1 and f3>=0 and f3<=1 and pa>=0 and pa<360:
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

def main(specifyThreads = False, save = True):
    if specifyThreads == False:
        threads = cpu_count() #for Summit job use all of them
    else:
        threads = 4 #for my computer
    print("running with {} threads".format(threads))
    data = readPickle("3c273_juljanmarmay_append_gilles_specirf_wide_v6.p")
    λCen = 2.172
    vel = (data[0]-λCen)/λCen*3e5
    pert = [0.3,10.,0.05,0.05,0.01,0.01,0.01,3.,] #i,rMin,Mfac,rFac,f1,f2,f3, pa, try for 0.5% around initial guess
    nWalkers = 50; θ0 = [30.,1e3,1.1,1.,0.57,0.6,0.46,342.] #i,rMin,Mfac,rFac,f1,f2,f3,pa
    p0 = numpy.zeros((len(θ0),nWalkers))
    for n in range(len(pert)):
        p0[n] = [θ0[n]+pert[n]*numpy.random.randn(1) for j in range(nWalkers)]
    p0 = numpy.transpose(p0)
    sampler,pos,prob,state = MC(nWalkers,θ0,p0,log_prob,vel,data,threads)
    flat_samples = sampler.get_chain(flat=True)
    if save == True: #ie on summit
        with open("pyEmceeVar.p","wb") as f:
            obj = [flat_samples,pos,prob] #really only care about flat_samples, don't think we need sampler/state
            pickle.dump(obj,f)
    else: #ie for interactive nb
        return sampler,pos,prob,state,flat_samples
