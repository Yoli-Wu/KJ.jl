function fitBlanks!(pd::run;n=2)
    channels = getChannels(pd)
    b = blankData(pd,channels=channels)
    nc = size(channels,1)
    bpar = fill(0.0,n*nc)
    for i in eachindex(channels)
        bpar[(i-1)*n+1:i*n] = polyFit(t=b[:,1],y=b[:,i+2],n=n)
    end
    setBlankPars!(pd,bpar)
end
export fitBlanks!

function parseBPar(bpar::Union{Nothing,AbstractVector{<:AbstractFloat}};par="bx")
    nbp = Integer(size(bpar,1)//3)
    if (par=="bx") return bpar[1:nbp]
    elseif (par=="by") return bpar[nbp+1:2*nbp]
    elseif (par=="bz") return bpar[2*nbp+1:3*nbp]
    else return nothing
    end
end
