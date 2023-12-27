# INCLUDE the WaterLily routine 
filePath = @__FILE__
workdir = dirname(filePath)
waterlilypath = dirname(dirname(workdir))*"/src/WaterLily.jl"
include(waterlilypath)

using .WaterLily
include("allCases.jl")
include("allUtils.jl")

function runCase(simset,cID,N,;dur=10,stp=0.01)

    sim = simset(N;T=Float64,Re=10000,λρ=1e-3,λμ=1e-2)

    puresim!(sim,dur,stp,cID)
end

function postProcess(cID)
    UScale, LScale, λρ, λμ, trueTime, NTime, dtTrueTime, dts, allTime, resIni, resFin, poisN, ke, Ng, Nv = readGeneralInfo(cID)
    VOFStore = zeros(Ng)
    VelStore = zeros(Nv)
    PreStore = zeros(Ng)
    VrtStore = zeros(Ng)
    KEnStore = zeros(Ng)

    KEList = zeros(NTime)

    cenTuple,edgTuple,limTuple = generateCoord(VOFStore)

    if length(Ng) == 2
        animGVOF = Plots.Animation()
        @time for iTime ∈ 1:NTime
            readData!(cID,iTime,VOFStore,VelStore,PreStore,UScale,LScale)
            calculateVort!(VrtStore,VelStore); VrtStore .*= LScale; WaterLily.BC!(VrtStore)
            KEList[iTime] = calculateKE!(KEnStore,VelStore,VOFStore,λρ)
            plt = Plots.plot(size=(1200,1000))
            plotContourf!(plt,edgTuple[1],edgTuple[2],VrtStore,clim=(-50,50))
            plotContour!(plt,cenTuple[1],cenTuple[2],VOFStore)
            organizePlot!(plt,limTuple[1],limTuple[2])
            frame(animGVOF,plt)
            iTime%10==1 && @printf("%04d/%04d\n", iTime, NTime)
        end
        Plots.gif(animGVOF,cID*"gVOF.gif",fps=50)
        plotKEEvole(cID,trueTime,dtTrueTime,KEList)
        analyze4KE(cID,ke,allTime,dts[1:end-1])
    elseif length(Ng) == 3
        @time for iTime ∈ 1:NTime
            readData!(cID,iTime,VOFStore,VelStore,PreStore,UScale,LScale)
            exportVelVOF(cID,iTime,VOFStore,VelStore,cenTuple)
            iTime%10==1 && @printf("%04d/%04d\n", iTime, NTime)
        end
    end
    plotPoisson(cID,allTime,dts,resIni,resFin,poisN)
end

function main()
    simu,pp = determineMode(ARGS)

    simset = sineWave2D
    N=64
    cID = setupCompID(getFunctionName(simset),N)

    simu && runCase(simset,cID,N,dur=2.3,stp=0.01)
    pp && postProcess(cID)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end




