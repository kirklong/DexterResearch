#!/usr/bin/env julia

using Statistics, PyCall, MPI, SimplePCHIP, JLD, Optim

scipyStats = pyimport("scipy.stats"); binnedStat = scipyStats.binned_statistic
@pyimport pickle

function readPickle(file)
    data = nothing
    @pywith pybuiltin("open")(file,"rb") as f begin
        data = pickle.load(f,encoding="latin1")
    end
    return data
end

function dvldl(r,sini,φ,windWeight=0,f1=1,f2=1,f3=1) #line of sight velocity gradient new way, highly dependent on H/R + i
    windφ1 = 3 .*sqrt.(1 ./(2 .*r))./r .*sini^2 .* cos.(φ).*(√2 .*cos.(φ).+sin.(φ)./2)
    windφ2 = cos(asin(sini))^2 .* (1 ./ (r.^(3/2))) # should be divided by H/R, so * R/H and say H/R ~ 0.01? leaving out for now because Jason said to
    windφ3 = -3 .*sqrt.(1 ./(2 .*r))./r .* sini*cos(asin(sini)) #./ r
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
    return α,β,r,ν,φ,sini
end

function getIntensity(r,φ,windWeight,sini,rMin=1e3,γ=1,A0=1,τ=10; f1=1,f2=1,f3=1)
    φn = φ .+ π/2 #+π/2 because we are different than CM by 90 deg
    ∇v = dvldl(r,sini,φn,windWeight,f1,f2,f3)
    A = getA(A0,r,γ)
    #calculate intensities
    I = intensity(A,r,∇v,τ); I[r.<rMin] .= 0.
    return I,γ,A0,τ
end

function phase(ν,I,x,y,U,V,bins=100)
    dφMap = @. -2*π*(x*U+y*V)*I*180/π*1e6 #1e6 is units of u,v 180/π to convert rad to deg, gives us corresponding dφ at every ν bin
    dφ,edges,n = binnedStat(vec(ν),vec(dφMap),statistic="sum",bins=bins) #phase binned along ν
    iSum,edges,n = binnedStat(vec(ν),vec(I),statistic="sum",bins=bins) #binned total I
    iSum[iSum.==0.].=1. #set to 1 in places beneath rMin so that we don't divide by 0
    return dφ./iSum
end

function getProfiles(ν,params,data=data) #incredibly inefficient to calculate the full thing at each step, but otherwise need to go back to beginning and setup differently?
    i,rMin,Mfac,rFac,windWeight,f1,f2,f3 = params
    blRange=Mfac*3e8*2e33*6.67e-8/9e20/548/3.09e24
    α,β,r,νloc,φ,sini = setup(i) #this ν is discrete, we will interpolate to make it continuous for fitting
    I,γ,A0,τ = getIntensity(r,φ,windWeight,sini,rMin,f1=f1,f2=f2,f3=f3)
    flux,νEdges,n = binnedStat(vec(νloc),vec(I),statistic="sum",bins=100) #only works with 100 bins idk why?
    νBin = 0.5*(νEdges[2:end].+νEdges[1:end-1]) #still discrete
    UData=data[2]; VData=data[3]; scipyNDImg = pyimport("scipy.ndimage"); G1D = scipyNDImg.gaussian_filter1d; dφList = []; psf=4e-3/2.35 #idk why this psf
    X=reshape(α.*blRange,(2048,2048)); Y=reshape(β.*blRange,(2048,2048));
    for i=1:length(UData)
        for ii in [I]
            dφAvgRaw=phase(νloc,ii,X,Y,UData[i],VData[i]) #phase(nu,ii,x,y,u[i],v[i],bins=bins)
            dφAvg = G1D(dφAvgRaw,psf/3e5/(νBin[2]-νBin[1])) #why psf/3e5/Δν ?
            push!(dφList,dφAvg)
        end
    end
    fline = flux./maximum(flux)*0.6./(1 .+ flux./maximum(flux).*0.6)
    indx=[0,1,2,6,7,8,12,13,14,18,19,20].+1
    x = (νBin.-1).*3e5; yP = mean(dφList[indx,:],dims=1)[1].*fline*rFac
    interpPhase = SimplePCHIP.interpolate(x,yP) #returns functional form of line profile from interpolation
    yL = flux./maximum(flux).*maximum(data[4])
    interpLine = SimplePCHIP.interpolate(x,yL) #returns functional form of line profile from interpolation
    return interpLine,interpPhase #this is the interpolated value as a fx of any ν, so we can match precisely to data
end

function getR(ν,i,rMin,Mfac,rFac=1.3,f1=1,f2=1,f3=1,windWeight=1,data=data)#get residuals from phase and line fit
    line,phase = getProfiles(ν,[i,rMin,Mfac,rFac,windWeight,f1,f2,f3],data)
    indx=[0,1,2,6,7,8,12,13,14,18,19,20].+1
    lRes = sqrt.((line.(ν).-data[4]).^2); phaseRes = sqrt.((phase.(ν).-mean(data[5][indx,:],dims=1)').^2)
    res = lRes .+ phaseRes
    return sum(res),sum(lRes),sum(phaseRes),line,phase
end

function main()
    println("starting optimization")
    data = readPickle("3c273_juljanmarmay_append_gilles_specirf_wide_v6.p")
    λCen = 2.172
    vel = (data[1].-λCen)./λCen.*3e5
    function f2Min(x,ν=vel,data=data)
        i,rMin,Mfac,rFac,f1,f2,f3 = x
        windWeight = 1
        R,Rline,Rphase,line,phase = getR(ν,i,rMin,Mfac,rFac,f1,f2,f3,windWeight,data)
        return R
    end
    x0 = [45,1e3,1,1.3,0.5,0.5,0.5]
    lower = [0.,500.,0.,0.,0.,0.,0.]; upper = [90.,1e4,100.,100.,1.,1.,1.]
    t0 = time()
    inner_optimizer = GradientDescent()
    result = optimize(f2Min,lower,upper,x0,Fminbox(inner_optimizer)) #takes like ~ 5-10 hours
    tf = time()
    println("finished optimization, saving results")
    println("total time elapsed = $(round((tf-t0)/3600,sigdigits=3)) hours")
    save("optimResult.jld","result",result)
end
