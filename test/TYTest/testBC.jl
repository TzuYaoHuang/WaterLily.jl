include("../../src/WaterLily.jl")

Ng = (6,7)

X = transpose(reshape([i+2.5*j for i=1:Ng[1] for j=1:Ng[2]],Ng[2],Ng[1]))
print(X)

WaterLily.BCPer!(X)
print(X)