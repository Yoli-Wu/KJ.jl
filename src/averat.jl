"""
averat

Average the 'atomic' isotopic ratios for a sample

# Returns

- a dataframe of P/D and d/D-ratios with their standard errors and error correlations

# Arguments

See [`process!`](@ref).
"""
function averat(run::Vector{Sample},
                channels::AbstractDict,
                blank::AbstractDataFrame,
                pars::NamedTuple;
                method=nothing)
    ns = length(run)
    if isnothing(method)
        xlab = "x"
        ylab = "y"
    else
        P, D, d = getPDd(method)
        xlab = P * "/" * D
        ylab = d * "/" * D
    end
    
    if method == "Rb-Sr"
        column_names = ["name", xlab, "s[" * xlab * "]", ylab, "s[" * ylab * "]", "rho",
                    "Rb87/Sr86", "s[Rb87/Sr86]", "Sr87/Sr86", "s[Sr87/Sr86]"]
        out = DataFrame(hcat(fill("", ns), zeros(ns, 9)), column_names)
    else
        column_names = ["name", xlab, "s[" * xlab * "]", ylab, "s[" * ylab * "]", "rho"]
        out = DataFrame(hcat(fill("", ns), zeros(ns, 5)), column_names)
    end
  
    for i in 1:ns
        samp = run[i]
        out[i,:name] = samp.sname
        x, sx, y, sy, rho = averat(samp, channels, blank, pars)
        
        if method == "Rb-Sr"
           Sr87_Sr86 = 1 / y
           s_Sr87_Sr86 = sy * (Sr87_Sr86^2)
           Rb87_Sr86 = x * Sr87_Sr86
           s_Rb87_Sr86 = Rb87_Sr86 * sqrt((sx / x)^2 + (s_Sr87_Sr86 / Sr87_Sr86)^2 + 2*rho*(sx / x)*(s_Sr87_Sr86 / Sr87_Sr86))
           out[i, 2:6] = [x, sx, y, sy, rho]
           out[i, 7:10] = [Rb87_Sr86, s_Rb87_Sr86, Sr87_Sr86, s_Sr87_Sr86]
       else
           out[i, 2:end] = [x, sx, y, sy, rho]
       end
    end
    return out
end

function averat(samp::Sample,
                channels::AbstractDict,
                blank::AbstractDataFrame,
                pars::NamedTuple)
    Phat, Dhat, dhat = atomic(samp,channels,blank,pars)
    return averat(Phat,Dhat,dhat)
end
function averat(Phat::AbstractVector,
                Dhat::AbstractVector,
                dhat::AbstractVector)
    vP = var_timeseries(Phat)
    vD = var_timeseries(Dhat)
    vd = var_timeseries(dhat)
    init = [sum(Phat)/sum(Dhat),sum(dhat)/sum(Dhat)]
    objective = (par) -> SSaverat(par[1],par[2],
                                  Phat,Dhat,dhat,
                                  vP,vD,vd)
    fit = Optim.optimize(objective,init)
    x, y = Optim.minimizer(fit)
    H = ForwardDiff.hessian(objective,[x,y])
    out = hessian2xyerr(H,[x,y])
    #E = covmat_averat(x,y,Phat,Dhat,dhat,vP,vD,vd)
    #out = [x sqrt(E[1,1]) y sqrt(E[2,2]) E[1,2]/sqrt(E[1,1]*E[2,2])]
    return out
end
export averat

function SSaverat(x::Real,
                  y::Real,
                  Phat::AbstractVector,
                  Dhat::AbstractVector,
                  dhat::AbstractVector,
                  vP::AbstractVector,
                  vD::AbstractVector,
                  vd::AbstractVector)
    D = averatD(x,y,Phat,Dhat,dhat,vP,vD,vd)
    return sum(@. (D*y-dhat)^2/vd+(D*x-Phat)^2/vP+(D-Dhat)^2/vD )/2
end

# block matrix inversion of the Hessian matrix
function covmat_averat(x::Real,
                       y::Real,
                       Phat::AbstractVector,
                       Dhat::AbstractVector,
                       dhat::AbstractVector,
                       vP::AbstractVector,
                       vD::AbstractVector,
                       vd::AbstractVector)
    D = averatD(x,y,Phat,Dhat,dhat,vP,vD,vd)
    O11 = sum(@. D^2/vP)
    O12 = O21 = 0.0
    O22 = sum(@. D^2/vd)
    O13 = @. (D*x-Phat)/vP+(D*x)/vP
    O23 = @. (D*y-dhat)/vd+(D*y)/vd
    O31 = @. ((2*(D*x-Phat))/vP+(2*D*x)/vP)/2
    O32 = @. ((2*(D*y-dhat))/vd+(2*D*y)/vd)/2
    O33 = @. ((2*y^2)/vd+(2*x^2)/vP+2/vD)/2
    H11 = [ [O11 O12]
            [O21 O22] ]
    H12 = [ O13' ; O23' ]
    H21 = [ O31 O32 ]
    H22 = diagm( O33 )
    return inv( H11 - H12 * inv(H22) * H21 )
end

function averatD(x,
                 y,
                 Phat::AbstractVector,
                 Dhat::AbstractVector,
                 dhat::AbstractVector,
                 vP::AbstractVector,
                 vD::AbstractVector,
                 vd::AbstractVector)
    return @. (dhat*vD*vP*y+Phat*vD*vd*x+Dhat*vP*vd)/(vD*vP*y^2+vD*vd*x^2+vP*vd)
end
