#!/usr/bin/env julia
using Statistics, PyCall, MPI, SimplePCHIP, JLD, Distributed

scipyStats = pyimport("scipy.stats"); binnedStat = scipyStats.binned_statistic
mc = pyimport("emcee")
scipyNDImg = pyimport("scipy.ndimage"); G1D = scipyNDImg.gaussian_filter1d;
@pyimport pickle; @pyimport multiprocessing

function readPickle(file)
    data = nothing
    @pywith pybuiltin("open")(file,"rb") as f begin
        data = pickle.load(f,encoding="latin1")
    end
    return data
end

function dvldl(r,sini,cosi,φ,windWeight=0,f1=1,f2=1,f3=1) #line of sight velocity gradient new way, highly dependent on H/R + i
    windφ1 = 3 .*sqrt.(1 ./(2 .*r))./r .*sini^2 .* cos.(φ).*(√2 .*cos.(φ).+sin.(φ)./2)
    windφ2 = cosi^2 .* (1 ./ (r.^(3/2))) # should be divided by H/R, so * R/H and say H/R ~ 0.01? leaving out for now because Jason said to
    windφ3 = -3 .*sqrt.(1 ./(2 .*r))./r .* sini*cosi #./ r
    diskφ = 3 .*sqrt.(1 ./(2 .*r))./r .*sini^2 .* (cos.(φ).*sin.(φ)./2) #disk only
    dvl = (1-windWeight).*diskφ .+ windWeight.*(f1.*windφ1 .+ f2.*windφ2 .+ f3.*windφ3)#.+ windφ3 #new terms approach, only doing first and last term for now because unclear how to do θ bit (significant re-write i think)
    return dvl #there is also a mass dependence embedded here in v_φ term we should insert for fit -- no, do later because this is all in r_g
end

getA(A0,x,γ) = A0.*x.^γ

intensity(A,r,∇v,τ) = A./(4*π.*r.^2).*abs.(∇v).*(1-exp(-τ))

function setup(i=75,nx=2048,ny=2048,rlim=3e4)
    #set up "camera" with coordinates (x,y) = α,β ; inclined 75 deg from pole
    a = range(0,stop=nx-1,length=nx)./(nx-1)*rlim.-rlim/2; b = range(0,stop=ny-1,length=ny)./(ny-1)*rlim.-rlim/2

    meshgrid(x,y) = (repeat(x,outer=length(y)), repeat(y,inner=length(x)))
    α,β = meshgrid(a,b)

    i = i/180*π; cosi = cos(i); sini = sin(i) #inclination angle in rad

    # calculate the raal and azimuthal coordinates wher rays sent from camera pixels intersect the "disk" in the equatorial plane, working back from camera
    r = reshape(sqrt.(β.^2 ./cosi^2 .+ α.^2),nx,ny); φ = reshape(atan.(β,α.*cosi),nx,ny)
    ν = 1 .+ sqrt.(1 ./(2 .*r)).*sini.*cos.(φ) #Doppler shift G = M = c = 1; unclear why 2?
    return α,β,r,ν,φ,sini,cosi
end

function getIntensity(r,φ,windWeight,sini,cosi,rMin=1e3,γ=1,A0=1,τ=10; f1=1,f2=1,f3=1)
    φn = φ .+ π/2 #+π/2 because we are different than CM by 90 deg
    ∇v = dvldl(r,sini,cosi,φn,windWeight,f1,f2,f3)
    A = getA(A0,r,γ)
    #calculate intensities
    I = intensity(A,r,∇v,τ); I[r.<rMin] .= 0.
    return I,γ,A0,τ
end

function phase(ν,I,x,y,U,V,bins=100)
    #shape of I is the problem here
    dφMap = @. -2*π*(x*U+y*V)*I*180/π*1e6 #1e6 is units of u,v 180/π to convert rad to deg, gives us corresponding dφ at every ν bin
    dφ,edges,n = binnedStat(vec(ν),vec(dφMap),statistic="sum",bins=bins) #phase binned along ν
    iSum,edges,n = binnedStat(vec(ν),vec(I),statistic="sum",bins=bins) #binned total I
    iSum[iSum.==0.].=1. #set to 1 in places beneath rMin so that we don't divide by 0
    return dφ./iSum
end

function getProfiles(ν,params,data=data,bins=100; nx=2048,ny=2048) #incredibly inefficient to calculate the full thing at each step, but otherwise need to go back to beginning and setup differently?
    i,rMin,Mfac,rFac,f1,f2,f3 = params; windWeight = 1
    blRange=Mfac*3e8*2e33*6.67e-8/9e20/548/3.09e24
    α,β,r,νloc,φ,sini,cosi = setup(i,nx,ny) #this ν is discrete, we will interpolate to make it continuous for fitting
    I,γ,A0,τ = getIntensity(r,φ,windWeight,sini,cosi,rMin,f1=f1,f2=f2,f3=f3)
    flux,νEdges,n = binnedStat(vec(νloc),vec(I),statistic="sum",bins=100) #only works with 100 bins idk why?
    νBin = 0.5*(νEdges[2:end].+νEdges[1:end-1]) #still discrete
    UData=data[2]; VData=data[3]; psf=4e-3/2.35 #idk why this psf
    X=reshape(α.*blRange,(nx,ny)); Y=reshape(β.*blRange,(nx,ny));
    dφList = Array{typeof(Vector{Float64}(undef,bins)),1}(undef,length(UData)*length(I)); ind = 1
    for i=1:length(UData)
        for ii in [I]
            dφAvgRaw=phase(νloc,ii,X,Y,UData[i],VData[i],bins) #phase(nu,ii,x,y,u[i],v[i],bins=bins)
            dφAvg = G1D(dφAvgRaw,psf/3e5/(νBin[2]-νBin[1])) #why psf/3e5/Δν ?
            dφList[ind] = dφAvg; ind += 1
        end
    end
    fline = flux./maximum(flux)*0.6./(1 .+ flux./maximum(flux).*0.6)
    indx=[0,1,2,6,7,8,12,13,14,18,19,20].+1; oindx=[3,4,5,9,10,11,15,16,17,21,22,23].+1
    x = (νBin.-1).*3e5; yP = mean(dφList[indx,:],dims=1)[1].*fline*rFac; yPo = mean(dφList[oindx,:],dims=1)[1].*fline*rFac
    interpPhase = SimplePCHIP.interpolate(x,yP); interpPhaseo = SimplePCHIP.interpolate(x,yPo) #returns functional form of mean phase profile in on vs off positions from interpolation
    yL = flux./maximum(flux).*maximum(data[4])
    interpLine = SimplePCHIP.interpolate(x,yL) #returns functional form of line profile from interpolation
    return interpLine,interpPhase, interpPhaseo #this is the interpolated value as a fx of any ν, so we can match precisely to data
end


function log_lhood(θ,x,data=data) #where θ are params, x is vel, data includes y/yErr
    lineModel,phaseModel,phaseoModel = getProfiles(x,θ,data)
    yLErr = data[7]; yPErr = x.*0.0.+0.07
    indx=[0,1,2,6,7,8,12,13,14,18,19,20].+1; oindx=[3,4,5,9,10,11,15,16,17,21,22,23].+1
    lnLikeLine = -(1/2) * sum(((data[4] .- lineModel.(x))./yLErr).^2)
    lnLikePhase = -(1/2) * sum(((mean(data[5][indx,:],dims=1)' .- phaseModel.(x))./yPErr).^2)
    lnLikePhaseo = -(1/2) * sum(((mean(data[5][oindx,:],dims=1)' .- phaseoModel.(x))./yPErr).^2)
    return lnLikeLine + lnLikePhase + lnLikePhaseo
end

function log_prior(θ)
    i,rMin,Mfac,rFac,f1,f2,f3 = θ
    if i > 0 && i < 90 && rMin > 500 && rMin < 1e4 && Mfac > 0 && rFac > 0 && f1 >= 0 && f1 <= 1 && f2 >= 0 && f2 <= 1 && f3 >= 0 && f3 <= 1
        return 0.0
    else
        return -Inf
    end
end

function log_prob(θ,x,data=data)
    lnP = log_prior(θ)
    if lnP == -Inf
        return -Inf
    else
        return lnP + log_lhood(θ,x,data)
    end
end

function pyMC(nWalkers,θ0,p0,log_prob,vel,data,threads,iter=100) #needed to wrap in function to prevent it from getting mad?
    if threads > 1 #this doesn't work
        @pywith multiprocessing.Pool(threads) as pool begin
            sampler = mc.EnsembleSampler(nWalkers,length(θ0),log_prob,args=(vel,data),pool=pool)
            println("running burn-in")
            p0, _, _ = sampler.run_mcmc(p0,10,progress=true,skip_initial_state_check=true)
            sampler.reset()
            println("production run (n = $iter iterations)")
            pos, prob, state = sampler.run_mcmc(p0,iter,skip_initial_state_check=true,progress=true)
            return sampler,pos,prob,state
        end
    else
        sampler = mc.EnsembleSampler(nWalkers,length(θ0),log_prob,args=(vel,data))
        println("running burn-in")
        p0, _, _ = sampler.run_mcmc(p0,100,progress=true,skip_initial_state_check=true)
        sampler.reset()
        println("production run (n = $iter iterations)")
        pos, prob, state = sampler.run_mcmc(p0,iter,skip_initial_state_check=true,progress=true)
        return sampler,pos,prob,state
    end
end

function main()
    threads = Threads.nthreads()
    println("starting up with $threads threads")
    @everywhere begin
        data = readPickle("3c273_juljanmarmay_append_gilles_specirf_wide_v6.p")
        λCen = 2.172
        vel = (data[1].-λCen)./λCen.*3e5
        pert = [1.,100.,0.1,0.1,0.1,0.1,0.1] #initial perturbations
        nWalkers = 50; θ0 = [45.,1e3,1.,1.3,0.5,0.5,0.5]
        p0 = zeros(nWalkers,length(θ0))
        for n=1:length(θ0)
            p0[:,n] = [θ0[n] + pert[n]*rand() for j=1:nWalkers]
        end
    end
    sampler,pos,prob,state = pyMC(nWalkers,θ0,p0,log_prob,vel,data,threads)
    println("saving emcee variables")
    save("jPyMC.jld","sampler",sampler,"pos",pos,"prob",prob,"state",state)
    println("flattening chains and saving")
    flat_samples = sampler.get_chain(flat=True)
    save("jPyMCFLAT.jld","flat_samples",flat_samples)
    println("exiting on successful completion")
end
