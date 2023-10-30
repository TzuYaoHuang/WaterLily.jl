using Plots


mutable struct aBubble
    CIs :: Vector{CartesianIndex}
    vol
    r
    cen
    function aBubble(ci)
        new(CartesianIndex[ci],0.0,0.0,zeros(length(ci)))
    end
end

struct BubblesInfo 
    labelField::AbstractArray{Int}
    bubbleDict::Dict{Int,aBubble}
    labelCount::Vector{Int}
    function BubblesInfo(f)
        f .= 0
        new(
            f, 
            Dict{Int,aBubble}(),
            [0]
        )
    end
end

@inline inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))

function negI(i,I,N)
    if I[i] != 2
        return CartesianIndex(ntuple(j -> j==i ? I[j]-1 : I[j], length(N)))
    else
        return CartesianIndex(ntuple(j -> j==i ? N[j]-1 : I[j], length(N)))
    end
end

function InitilizeBubbleInfo!(bInfo::BubblesInfo)
    bInfo.labelField .= 0
    empty!(bInfo.bubbleDict)
    bInfo.labelCount .= 0
end


function getLabel(p,l)
    return l[p]
end
getLabel(p, bInfo::BubblesInfo) = getLabel(p, bInfo.labelField)


function mergeBubbles!(bDict,labelField,labels)
    # Merge in bubble Dict
    targetL = minimum(labels)
    for l ∈ labels
        if l != targetL
            append!(bDict[targetL].CIs,bDict[l].CIs)
            delete!(bDict, l)   # Delete entry
        end
    end
    # Merge in label field
    for I ∈ bDict[targetL].CIs
        labelField[I] = targetL
    end
end
mergeBubbles!(bInfo::BubblesInfo, labels) = mergeBubbles!(bInfo.bubbleDict, bInfo.labelField, labels)

function assignLabel!(bDict,labelField,p,label)
    # Assign in bubble Dict
    push!(bDict[label].CIs,p)
    # Assign in labelField
    labelField[p] = label
end
assignLabel!(bInfo::BubblesInfo,p,label) = assignLabel!(bInfo.bubbleDict, bInfo.labelField,p,label)

function newLabel!(bDict,labelField,labelCount,p)
    labelCount .+= 1
    # new in bubble Dict
    bDict[labelCount[1]] = aBubble(p)
    # new in labelField
    labelField[p] = labelCount[1]
end
newLabel!(bInfo::BubblesInfo,p) = newLabel!(bInfo.bubbleDict, bInfo.labelField, bInfo.labelCount, p)

function InformedCCL!(bInfo,v,θ,n̂,I;useNormConnect=true)
    normConnect = true
    N = size(v)
    n = length(N)
    if v[I] > θ
        p = I
        for d ∈ 1:n
            pᵖ = negI(d,I,N)
            pl = getLabel(p,bInfo)
            pᵖl = getLabel(pᵖ,bInfo)

            if useNormConnect
                normConnect = (n̂[p,d]*n̂[pᵖ,d]>0)
                if (n̂[pᵖ,d]<0) && (n̂[p,d]>0)
                    normConnect = true
                end
                if (n̂[pᵖ,d]<0) && (v[p]==1)
                    normConnect = true
                end
                if (n̂[p,d]>0) && (v[pᵖ]==1)
                    normConnect = true
                end
                if (v[p]==1) && (v[pᵖ]==1)
                    normConnect = true
                end
            end
            
            if (pᵖl==0) && (pl==0)
                newLabel!(bInfo,p)
            end
            if (pᵖl!=0) && (pl==0) && !normConnect
                newLabel!(bInfo,p)
            end
            if (pᵖl!=0) && (pl==0) && normConnect
                assignLabel!(bInfo,p,pᵖl)
            end
            if (pᵖl!=0) && (pl!=0) && (pᵖl!=pl) && normConnect
                mergeBubbles!(bInfo,[pᵖl,pl])
            end
        end
    end
end

function CalculateBubbleInfo!(bInfo::BubblesInfo,v::AbstractArray{T,D}) where {T,D}
    for (label,bubble) ∈ bInfo.bubbleDict
        vol = 0
        cen = zeros(length(bubble.CIs[1]))
        for I ∈ bubble.CIs
            vol += v[I]
            @. cen += (I.I - 1.5)*vol
        end
        cen ./= vol
        bubble.vol = vol
        bubble.r = D==3 ? (3/4*vol/π)^(1/3) : (vol/π)^(1/2)
        bubble.cen = cen
    end
end

function ProjectBubbleInfoToVOFFiled!(bInfo::BubblesInfo)
    bInfo.labelField .= 0
    for (label,bubble) ∈ bInfo.bubbleDict
        for I ∈ bubble.CIs
            if bInfo.labelField[I]>0
                error("SOME OVERLAP!!!")
            end
            bInfo.labelField[I] = label
        end
    end
end

function ICCL_M!(bInfo,v,θs,n̂;useNormConnect=true)
    for θ ∈ θs
        for I ∈ inside(v)
            InformedCCL!(bInfo,v,θ,n̂,I,useNormConnect=useNormConnect)
        end
    end
    CalculateBubbleInfo!(bInfo,v)
    return bInfo
end
