include("../../src/WaterLily.jl")

using Plots
using Statistics


optPois = "Pois"
optSour = "Cylinder"
optBCon = "Per"

stepbystep = false

computationID = optPois * "_" * optSour * "_" * optBCon * "_"
print(computationID,"\n")

T=Float32
N = (10*2^7,2^7)
D = length(N)
Ng = N .+ 2
Nd = (Ng..., 2)

l = N[2]
xc, yc = N[1]/2,N[2]/2
R = 2^4
U=1
nSum = 10

function SourceFunc(x,y)
    r = sqrt.((x.-xc).^2+(y.-yc).^2)

    return -32*pi^6*R^4*U^2/l^6*(
        (cosh.(2*pi*(x.-xc)/l).+cos.(2*pi*(y.-yc)/l))./
        (cosh.(2*pi*(x.-xc)/l).-cos.(2*pi*(y.-yc)/l)).^3
    ).*(r .> 0.9R)
end

function ManufacSol(x,y)
    r = sqrt.((x.-xc).^2+(y.-yc).^2)

    return 2*R^2*pi^2*U^2/l^4*(
        (l^2*cosh.(2*pi*(x.-xc)/l).*cos.(2*pi*(y.-yc)/l).-pi^2*R^2 .-l^2)./
        (cosh.(2*pi*(x.-xc)/l).-cos.(2*pi*(y.-yc)/l)).^2
    ).*(r .> 0.9R)
end

body = WaterLily.AutoBody((x,t)->√sum(abs2, x .- Array([xc,yc])) - R)
flow = WaterLily.Flow(N,(0,0))
flow.μ₀ .= 1.0
WaterLily.measure!(flow,body;ϵ=1)
μ₀ = flow.μ₀


#optPois=="MulP" && WaterLily.BC!(μ₀,ntuple(zero, D))



X = transpose(reshape([i-1.5 for i=1:Ng[1] for j=1:Ng[2]],Ng[2],Ng[1]))
Y = transpose(reshape([j-1.5 for i=1:Ng[1] for j=1:Ng[2]],Ng[2],Ng[1]))


source = SourceFunc(X,Y)
realSol = ManufacSol(X,Y)
originalSource = copy(source)
solIni = similar(realSol)

# BCPer!(solIni)
WaterLily.BCPer!(source)
# BCPerVec!(μ₀)


optPois=="Pois" && (pTest = WaterLily.Poisson(solIni, μ₀, source))
optPois=="MulP" && (pTest = WaterLily.MultiLevelPoisson(solIni, μ₀, source))




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
r = sqrt.((X.-xc).^2+(Y.-yc).^2)
pTest.x .*= (r .> 0.9R)
pTest.x .-= mean(pTest.x[2,2:end-1]) 
realSol .-= mean(realSol[2,2:end-1])


plot(res)
plot!(yscale=:log10, minorgrid=true)
savefig(computationID * "Residual.png")


plot(contourf(X[2:end-1,2:end-1],Y[2:end-1,2:end-1],pTest.x[2:end-1,2:end-1];levels=32,aspect_ratio = :equal), size = (2400,600))
savefig(computationID * "Calculated.png")


plot(contourf(X[2:end-1,2:end-1],Y[2:end-1,2:end-1],log10.(abs.(realSol[2:end-1,2:end-1]-pTest.x[2:end-1,2:end-1]).+1e-12);levels=32,aspect_ratio = :equal), size = (2400,600))
savefig(computationID * "Error.png")


plot(contourf(X[2:end-1,2:end-1],Y[2:end-1,2:end-1],realSol[2:end-1,2:end-1];levels=32,aspect_ratio = :equal), size = (2400,600))
savefig(computationID * "Correct.png")


plot(contourf(X[2:end-1,2:end-1],Y[2:end-1,2:end-1],originalSource[2:end-1,2:end-1];levels=32,aspect_ratio = :equal), size = (2400,600))
savefig(computationID * "Source.png")


