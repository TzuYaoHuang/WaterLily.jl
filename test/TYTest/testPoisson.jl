include("../../src/WaterLily.jl")

using Plots
using Statistics


optPois = "Pois"
optSour = "Hard"
optBCon = "Per"

stepbystep = false

computationID = optPois * "_" * optSour * "_" * optBCon * "_"
print(computationID,"\n")

T=Float32
N = (2^7,3*2^7)
D = length(N)
Ng = N .+ 2
Nd = (Ng..., 2)

offset = -0.3


optSour=="Hard" && (SourceFunc(x,y) = (
    -4 *pi^2 * (1/N[1]^2 + 1/N[2]^2) * sin.(2*pi*(x/N[1].+offset)) .* sin.(2*pi*(y/N[2].+offset)) +
    -16*pi^2 * (1/N[1]^2 + 1/N[2]^2) * sin.(8*pi*(x/N[1].+offset)) .* sin.(8*pi*(y/N[2].+offset))
))

optSour=="Hard" && (ManufacSol(x,y) = (
    sin.(2*pi*(x/N[1].+offset)) .* sin.(2*pi*(y/N[2].+offset)) +
    sin.(8*pi*(x/N[1].+offset)) .* sin.(8*pi*(y/N[2].+offset))/4
))

optSour=="Easy" && (SourceFunc(x,y) = (
    4*pi^2 * (
        cos.(2*pi*(x/N[1])) .* (1 .-cos.(2*pi*(y/N[2])))/N[1]^2 +
        (1 .-cos.(2*pi*(x/N[1]))) .* cos.(2*pi*(y/N[2]))/N[2]^2
    )
))

optSour=="Easy" && (ManufacSol(x,y) = (
    (1 .-cos.(2*pi*(x/N[1]))) .* (1 .-cos.(2*pi*(y/N[2]))) 
))


waveNum = 4*pi
optSour=="Middle" && (SourceFunc(x,y) = (
    waveNum^2 * (
        cos.(waveNum*(x/N[1])) .* (1 .-cos.(waveNum*(y/N[2])))/N[1]^2 +
        (1 .-cos.(waveNum*(x/N[1]))) .* cos.(waveNum*(y/N[2]))/N[2]^2
    )
))

optSour=="Middle" && (ManufacSol(x,y) = (
    (1 .-cos.(waveNum*(x/N[1]))) .* (1 .-cos.(waveNum*(y/N[2]))) 
))


solIni = zeros(Ng)
source = zeros(Ng)
μ₀ = ones(Nd)
#optPois=="MulP" && WaterLily.BC!(μ₀,ntuple(zero, D))

# BCPer!(solIni)
# BCPer!(source)
# BCPerVec!(μ₀)

X = transpose(reshape([i-1.5 for i=1:Ng[1] for j=1:Ng[2]],Ng[2],Ng[1]))
Y = transpose(reshape([j-1.5 for i=1:Ng[1] for j=1:Ng[2]],Ng[2],Ng[1]))


source = SourceFunc(X,Y)
realSol = ManufacSol(X,Y)
originalSource = copy(source)

realSol .-= mean(realSol[2:end-1,2:end-1])

optPois=="Pois" && (pTest = WaterLily.Poisson(solIni, μ₀, source))
optPois=="MulP" && (pTest = WaterLily.MultiLevelPoisson(solIni, μ₀, source))

res = []


stepbystep && for i ∈ 1:200
    partRes = WaterLily.solver!(pTest;log=true,tol=1e-12,itmx=20)
    # pTest.x .-= mean(pTest.x[2:end-1,2:end-1]) 
    pythonplot()
    contourf(X[2:end-1,2:end-1],Y[2:end-1,2:end-1],pTest.x[2:end-1,2:end-1];levels=32)
    savefig("bunchFig/" * computationID * "Calculated_"*string(i, pad=4)*".png")
    # pythonplot()
    # contourf(X[2:end-1,2:end-1],Y[2:end-1,2:end-1],log10.(abs.(realSol[2:end-1,2:end-1]-pTest.x[2:end-1,2:end-1]).+1e-12);levels=32)
    # savefig("bunchFig/" * computationID * "Error_"*string(i, pad=4)*".png")

    push!(res,partRes...)
end

!stepbystep && (res = WaterLily.solver!(pTest;log=true,tol=1e-12,itmx=5e3))

#print(res)


pTest.x .-= mean(pTest.x[2:end-1,2:end-1]) 
realSol .-= mean(realSol[2:end-1,2:end-1]) 

pythonplot()
plot(res)
plot!(yscale=:log10, minorgrid=true)
savefig(computationID * "Residual.png")

pythonplot()
contourf(X[2:end-1,2:end-1],Y[2:end-1,2:end-1],pTest.x[2:end-1,2:end-1];levels=32)
savefig(computationID * "Calculated.png")

pythonplot()
contourf(X[2:end-1,2:end-1],Y[2:end-1,2:end-1],log10.(abs.(realSol[2:end-1,2:end-1]-pTest.x[2:end-1,2:end-1]).+1e-12);levels=32)
savefig(computationID * "Error.png")

pythonplot()
contourf(X[2:end-1,2:end-1],Y[2:end-1,2:end-1],realSol[2:end-1,2:end-1];levels=32)
savefig(computationID * "Correct.png")

pythonplot()
contourf(X[2:end-1,2:end-1],Y[2:end-1,2:end-1],originalSource[2:end-1,2:end-1];levels=32)
savefig(computationID * "Source.png")


