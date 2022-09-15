using Debugger
include("DiskWind/src/functions.jl")

function setup2(i::Float64,n1::Int64,n2::Int64,r̄::Float64,rFac::Float64,rMin::Float64,rMax::Float64,coordsType::Symbol=:polar,scale::Symbol=:log)
    #rMin,rMax = get_rMinMax(r̄,rFac,γ)
    i = i/180*π; cosi = cos(i); sini = sin(i) #inclination angle in rad
    α = nothing; β = nothing; r = nothing; ν = nothing; ϕ = nothing; dA = nothing
    if coordsType == :cartesian
        nx = n1; ny = n2; rlim = rMax
        a = nothing; b = nothing

        if scale == :linear
            a = range(-rlim,stop=rlim,length=nx); b = range(-rlim,stop=rlim,length=ny)
        elseif scale == :log
            a = vcat(-reverse(exp.(range(log(rMin*cosi),stop=log(rMax),length=Int(nx/2)))),exp.(range(log(rMin*cosi),stop=log(rMax),length=Int(nx/2))))
            b = vcat(-reverse(exp.(range(log(rMin*cosi),stop=log(rMax),length=Int(ny/2)))),exp.(range(log(rMin*cosi),stop=log(rMax),length=Int(ny/2))))
        else
            println("invalid scale symbol -- should be :linear or :log")
            exit()
        end

        α,β = meshgrid(a,b)
        α = reshape(α,nx,ny); β = reshape(β,nx,ny)

        dA = zeros(size(α))
        for i=1:size(dA)[1]
            for j=1:size(dA)[2]
                Δα = i<size(dA)[1] ? abs(α[i+1,j]-α[i,j]) : abs(α[end,j]-α[end-1,j]) #kinda bs but want n things and linear spacing so fine
                Δβ = j<size(dA)[2] ? abs(β[i,j+1]-β[i,j]) : abs(β[i,end]-β[i,end-1])
                dA[i,j] = Δα*Δβ
            end
        end

        r = reshape(sqrt.(β.^2 ./cosi^2 .+ α.^2),nx,ny); ϕ = reshape(atan.(β./cosi,α),nx,ny)
        ν = 1 .+ sqrt.(1 ./(2 .*r)).*sini.*cos.(ϕ)

    elseif coordsType == :polar
        nr = n1; nϕ = n2
####################################################
        #offset = rand()*0.01
        #offset = exp(0.01) #irrational number, consistent -- UPDATE: deprecated with new implementation of binning that centers bins on 0, fixes problem more elegantly
        offset = 0.
        ϕ = range(0+offset,stop=2π+offset,length=nϕ+1)[1:end-1] #so exclusive -- THIS IS THE PROBLEM
####################################################
        r = nothing; rGhost = nothing; Δr = nothing; Δlogr = nothing

        if scale == :linear
            r = range(rMin*cosi,stop=rMax,length=nr)
            Δr = r[2]-r[1]
            rGhost = [rMin*cosi-Δr,rMax*cosi+Δr]
        elseif scale == :log
            logr = range(log(rMin*cosi),stop=log(rMax),length=nr)
            Δlogr = logr[2]-logr[1]
            rGhost = exp.([log(rMin*cosi)-Δlogr,log(rMax)+Δlogr])
            r = exp.(logr)
        else
            println("invalid scale symbol -- should be :linear or :log")
            exit()
        end

        rMesh, ϕMesh = meshgrid(r,ϕ)
        rMesh = reshape(rMesh,nr,nϕ); ϕMesh = reshape(ϕMesh,nr,nϕ)
        α,β = rMesh.*cos.(ϕMesh), rMesh.*sin.(ϕMesh)
        #α,β = rMesh, ϕMesh
        Δϕ = ϕ[2]-ϕ[1]
        dA = zeros(size(rMesh))
        for i=1:size(dA)[1]
            for j=1:size(dA)[2]
                if scale == :log
                    #Δr = exp(Δlogr*(i-1))-exp(Δlogr*(i-2)) #assuming min r = 1. i.e. min logr = 0.
                    #jason says this should just be r*Δlogr -- calculus -- but doing it that way is not as good when I test error?
                    Δr = rMesh[i,j]*Δlogr
                end
                dA[i,j] = rMesh[i,j]*Δϕ*Δr
            end
        end

        r = reshape(sqrt.(β.^2/cosi^2 .+ α.^2),nr,nϕ); ϕ = reshape(atan.(β./cosi,α),nr,nϕ)
        ν = 1 .+ sqrt.(1 ./(2 .* r)).*sini.*cos.(ϕ)

    else
        println("invalid coords system -- should be :cartesian or :polar")
        exit()
    end
    return reshape(α,n1,n2),reshape(β,n1,n2),r,ν,ϕ,sini,cosi,dA,rMin,rMax
end

function mainCode(nR,nϕ,nBinv,nBint;tMaxR=20.,directCompare = false,binCenter = true,rMin=0.5e3,rMax=1e4,
    i=73.,rBar=6e3,M=7.8e7*2e30,rFac=45.,f1=0.57,f2=0.6,f3=0.38,f4=0.21,coords=:polar,scale=:log)
    start = time()
    G = 6.67e-11; c = 2.99e8; days =3.6e3*24.
    rs = 2*G*M/c^2
    α,β,rArr,nuArr,ϕArr,sini,cosi,dA,rMin,rMax = setup2(i,nR,nϕ,rBar,rFac,rMin,rMax,coords,scale)
    if directCompare == true
        α,β,rArr,nuArr,ϕArr,sini,cosi,dA,rMin,rMax = setup(i,nR,nϕ,rBar,rFac,1.,coords,scale)
    end
    ϕ′ = ϕArr .+ π/2 #waters ϕ
    tArr = rArr .* (rs/c/days) .* (1 .- cos.(ϕ′).*sini)
    yArr = -sqrt.(1 ./ (2 .*rArr)).*sin.(ϕ′).*sini

    #JCode = -sqrt.(1 ./ (2 .*rArr))./c .* sini .* ((1 .- 3 .* (cos.(ϕ′)).^2)./2 .* sini .+ cos.(ϕ′))
    ICode,γ,A0,τ = getIntensity(rArr,ϕArr,sini,cosi,rMin,rMax,1.,1.,10.,f1=f1,f2=f2,f3=f3,f4=f4,test=false,noAbs=false) #this does the Waters conversion within
    #ICode#.*= (c) #to get to units do c/rs *rs (vϕ/c/r * r)
    # if directCompare == true
    #     ICode .*= rArr #.* rs#get rid of r dependence, physical units
    #     #ICode .*= dA
    #     #ICode .*= rArr #cancel out the extra r in A to get just A0
    # end
    #ΨCode = ICode ./ abs.(JCode)
    ΨCode = ICode.*dA
    #ΨCode[ΨCode .== 0.] .= 1e-30

    # p=scatter(yArr[rArr .> rMin].* (c/1e6),tArr[rArr .> rMin],label="",markerz=log10.(ΨCode[rArr .> rMin]),markerstrokewidth=0.,markersize=1.,
    #     ylims=(0,20),size=(550,600),markercolor=cgrad([:white,:forestgreen,:black],[0.,-2.5,-5.0]))
    # png(p,"tmp2.png")

    yMax = maximum(yArr); yMin = minimum(yArr)
    tMax = maximum(tArr); tMin = minimum(tArr)
    yBinned = range(yMin,stop=yMax,length=nBinv);
    #tBinned1 = range(tMin,stop=tMaxR,length=nBint+1)[1:end-1];tBinned2=range(tMaxR,stop=tMax,length=64)
    #tBinned=vcat(tBinned1,tBinned2)
    tBinned=range(tMin,stop=tMaxR,length=nBint)
    Δy = yBinned[2]-yBinned[1]; Δt = tBinned[2]-tBinned[1]
    #Δt1 = tBinned1[2]-tBinned1[1]; Δt2 = tBinned2[2]-tBinned2[1]
    Δt = tBinned[2]-tBinned[1]
    ΨBinned = zeros(length(yBinned),length(tBinned))

    if binCenter == true
        Threads.@threads for i=1:length(yBinned)
            #print("$(round(i/length(yBinned)*100,sigdigits=3)) % complete \r")
            yMinTmp = yBinned[i]-Δy/2; yMaxTmp = yBinned[i] + Δy/2
            for j=1:length(tBinned)
                #Δt = j<nBint ? Δt1 : Δt2
                tMinTmp = tBinned[j]-Δt/2; tMaxTmp = tBinned[j] + Δt/2
                mask = (yArr .>= yMinTmp) .& (yArr .< yMaxTmp) .& (tArr .>= tMinTmp) .& (tArr .< tMaxTmp) .& (rArr .> rMin)
                s = sum(ΨCode[mask])#.*dA[mask])
                if s > 0
                    ΨBinned[i,j] = s#/length(ΨCode[mask])
                else
                    ΨBinned[i,j] = 1e-30
                end
            end
        end
    else
        ΨBinned = zeros(length(yBinned)-1,length(tBinned)-1)
        Threads.@threads for i=1:length(yBinned)-1
            #print("$(round(i/length(yBinned)*100,sigdigits=3)) % complete \r")
            yMinTmp = yBinned[i]; yMaxTmp = yBinned[i+1]
            for j=1:length(tBinned)-1
                tMinTmp = tBinned[j]; tMaxTmp = tBinned[j+1]
                mask = (yArr .>= yMinTmp) .& (yArr .< yMaxTmp) .& (tArr .>= tMinTmp) .& (tArr .< tMaxTmp) .& (rArr .> rMin)
                s = sum(ΨCode[mask])#).*dA[mask])
                if s > 0
                    ΨBinned[i,j] = s#/length(ΨCode[mask])
                else
                    ΨBinned[i,j] = 1e-30
                end
            end
        end
        yBinned = yBinned[1:end-1]; tBinned = tBinned[1:end-1] #only return left edges
    end
    LP = histSum(yArr,ICode.*dA,bins=nBinv,νMin=minimum(yArr),νMax=maximum(yArr))
    p=plot(LP[2].*(3e8/1e6),LP[3],label="",xlims=(-10,10))
    png(p,"LPtmp.png")
    A0=maximum(LP[3])
    println("A0 (LP) = $A0")
    println("rs = $rs")
    #A0 = maximum(ΨBinned)
    println("maximum Ψ = $(maximum(ΨBinned))")
    ΨBinned[ΨBinned .== 0] .= 1e-30 #get rid of zero values so there are no log errors
    ΨBinned = ΨBinned ./ A0 #A0 = max(Ψ)
    println("A0 = $A0")
    finish = time()
    println("took $(round((finish-start)/60,sigdigits=3)) min for (nR,nϕ,nBinv,nBint) = ($nR,$nϕ,$nBinv,$nBint)")
    return yBinned,tBinned,ΨBinned,ΨCode,ICode,dA,tArr,yArr,rArr,ϕArr
end


function getΨMatch(ΨBinned,levels=[-0.4*i for i=0:11])
    #match matplotlib default:
    #"fills intervals that are closed at the top;
    #that is, for regions z1 and z2 the filled region is z1 < Z <= z2"

    logΨ = log10.(ΨBinned)
    res = zeros(size(logΨ))
    mask = (logΨ .<= levels[2]) .& (logΨ .>= levels[1])
    res[mask] .= (levels[1]+levels[2])/2
    for i=2:length(levels)-1
        mask = (logΨ .< levels[i]) .& (logΨ .>= levels[i+1])
        res[mask] .= (levels[i]+levels[i+1])/2
    end
    mask = (logΨ .< levels[end])
    res[mask] = logΨ[mask]
    return res
end

function makePlot(yBinned,tBinned,ΨBinned,rs,c,stype=:heatmap;levels=[-0.4*i for i=0:11],size=(720,540),clims=(-5,0),A0=1,tlims=(0,40),vlims=(-12,12),
    plotLPExact=false)
    ΨDiscrete = getΨMatch(ΨBinned,levels)
    clrticks = ["$(round(level,sigdigits=2))" for level in reverse(levels)]
    n = length(levels)
    yt = range(0,1,n)[1:n] #.+ 0.5/n add the half if levels are centered

    l = @layout [
        [a{0.7w} b]
        [c{0.3h,0.7w} d]
        ]

    colors = palette([:white,:forestgreen,:black],length(levels)-1)
    p1 = plot(yBinned.*(3e8/1e6),tBinned[(tBinned.<=tlims[2]) .& (tBinned.>=tlims[1])],ΨDiscrete[:,(tBinned.<=tlims[2]) .& (tBinned.>=tlims[1])]',
        color=colors,cbar=false,tickfont="Computer Modern",guidefont="Computer Modern",
        xlims=vlims,ylims=tlims,seriestype=stype,fill=true,levels=n,xticks=([-10,-5,0,5,10],["","","","",""]),bottom_margin=0*Plots.Measures.mm,
        xlabel="",ylabel="t [days]",minorticks=true,tickdirection=:out,minorgrid=true,clims=clims,
        framestyle=:box,right_margin=0*Plots.Measures.mm)
    #p2 = plot([NaN], lims=(0,1), framestyle=:none, legendDorodnitsyn=false) -- if you want a colorbar title

    xx = range(0,1,100)
    zz = zero(xx)' .+ xx
    p1 = plot!(title="",inset=(1,bbox(1/40,1/10,0.1,0.5)),titlefont="Computer Modern")
    p1 = plot!(p1[2],xx, xx, zz, ticks=false, ratio=10, legend=false, fc=colors, lims=(0,1),title="logΨ",
             framestyle=:box, right_margin=20*Plots.Measures.mm,seriestype=:heatmap,cbar=false,titlefontsize=10)

    for (yi,ti) in zip(yt,clrticks)
        p1=plot!(p1[2],annotations=(1.5,yi,text(ti, 7, "Computer Modern")))
    end
    #[annotate!(1.5, yi, text(ti, 7, "Computer Modern")) for (yi,ti) in zip(yt,clrticks)]
    #p1 = plot!(p1[2],annotations=[])
    #annotate!(2.2,0.5,text("logΨ",10,"Computer Modern",rotation=90))

    #now make Ψ(τ)
    #dt1 = [tBinned[2]-tBinned[1] for i=1:nBint]; dt2 = [tBinned[end]-tBinned[end-1] for i=1:64]
    #dt = vcat(dt1,dt2)./(3600*24)
    Ψτ = [(sum(ΨBinned[:,i])) for i=1:length(tBinned)]./(rs/c) #for normalization make unitless again
    τ_mean = sum(tBinned.*Ψτ)/sum(Ψτ)
    p2=plot(Ψτ,tBinned,label="",lw=2,color=:crimson,xlabel="Ψ(t)",framestyle=:box,minorticks=true,minorgrid=true,ymirror=true,
    xflip=true,guidefont="Computer Modern",tickfont="Computer Modern",xlims=(0.,0.01),ylims=tlims,title="mean delay = $(round(τ_mean,sigdigits=3)) days",
    xrotation=90,bottom_margin=0*Plots.Measures.mm,left_margin=0*Plots.Measures.mm,tickdirection=:out,titlefont="Computer Modern",titlefontsize=10)

    #now make LP
    LP2 = [sum(ΨBinned[i,:]) for i=1:length(yBinned)]
    p3=plot(yBinned.*(3e8/1e6),LP2./maximum(LP2),label="",color=:crimson,lw=2,minorticks=true,minorgrid=true,framestyle=:box,guidefont="Computer Modern",
        tickfont="Computer Modern",xlims=vlims,xlabel="Δv [Mm/s]",widen=false,ylims=(0,1.1),ylabel="Normalized flux Ψ(ν)",
        top_margin=0*Plots.Measures.mm,right_margin=0*Plots.Measures.mm,tickdirection=:out,yticks=[0.2*i for i=0:5])
    if plotLPExact!=false
        p3=plot!(plotLPExact[2].*(3e8/1e6),plotLPExact[3]./maximum(plotLPExact[3]),label="exact",color=:dodgerblue)
    end
    #p4=plot(left_margin=0*Plots.Measures.mm,right_margin=0*Plots.Measures.mm,top_margin=0*Plots.Measures.mm,bottom_margin=0*Plots.Measures.mm)
    P=plot(p1, p2, p3,layout=l, margins=0*Plots.Measures.mm,size=size,link=:both)
    return P
end
