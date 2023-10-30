# get sample attributes from a run:
function accesSample(pd::run,
                     i::Union{Nothing,Integer,Vector{Integer}},
                     T::Type,
                     fun::Function)
    if isnothing(i) i = 1:length(pd) end
    samples = getSamples(pd)[i]
    if isa(i,Integer)
        out = fun(samples)
    else
        out = Vector{T}(undef,size(i,1))
        for j in eachindex(samples)
            out[j] = fun(samples[j])
        end
    end
    out
end
function accessSample!(pd::run,
                       i::Union{Integer,Vector{Integer}},
                       fun::Function,val::Any)
    samples = getSamples(pd)
    for j in i fun(samples[j],val) end
    setSamples!(pd,samples=samples)
end
# set the control parameters inside a run:
function accessControl!(pd::run,attribute::Symbol,fun::Function,val::Any)
    ctrl = getControl(pd)
    (ctrl,A)
    setControl(pd,ctrl=ctrl)
end

# get sample attributes
function getSname(pd::sample) getproperty(pd,:sname) end
function getDateTime(pd::sample) getproperty(pd,:datetime) end
function getDat(pd::sample) getproperty(pd,:dat) end
function getBWin(pd::sample) getproperty(pd,:bwin) end
function getSWin(pd::sample) getproperty(pd,:swin) end
function getStandard(pd::sample) getproperty(pd,:standard) end

# get run attributes
function getSamples(pd::run) getproperty(pd,:samples) end
function getControl(pd::run) getproperty(pd,:control) end
function getBPar(pd::run) getproperty(pd,:bpar) end
function getSPar(pd::run) getproperty(pd,:spar) end
function getBCov(pd::run) getproperty(pd,:bcov) end
function getSCov(pd::run) getproperty(pd,:scov) end

# get sample attributes from a run
function getSnames(pd::run;i=nothing) accesSample(pd,i,String,getSname) end
function getDateTimes(pd::run;i=nothing) accesSample(pd,i,DateTime,getDateTime) end
function getDat(pd::run;i=nothing) accesSample(pd,i,DataFrame,getDat) end
function getBWin(pd::run;i=nothing) accesSample(pd,i,Vector{window},getBWin) end
function getSWin(pd::run;i=nothing) accesSample(pd,i,Vector{window},getSWin) end
function getStandard(pd::run;i=nothing) accesSample(pd,i,Integer,getStandard) end

# get control attributes
function getA(ctrl::Union{Nothing,control}) return isnothing(ctrl) ? nothing : getproperty(ctrl,:A) end
function getB(ctrl::Union{Nothing,control}) return isnothing(ctrl) ? nothing : getproperty(ctrl,:B) end
function getChannels(ctrl::Union{Nothing,control}) return isnothing(ctrl) ? nothing : getproperty(ctrl,:channels) end

# get control attributes from a run
function getA(pd::run) getA(getControl(pd)) end
function getB(pd::run) getB(getControl(pd)) end
function getChannels(pd::run) getChannels(getControl(pd)) end

# set sample attributes
function setSname!(pd::sample;sname::String) setproperty!(pd,:sname,sname) end
function setDateTime!(pd::sample;datetime::DateTime) setproperty!(pd,:datetime,datetime) end
function setDat!(pd::sample;dat::DataFrame) setproperty!(pd,:dat,dat) end
function setBWin!(pd::sample,bwin::Vector{window}) setproperty!(pd,:bwin,bwin) end
function setSWin!(pd::sample,swin::Vector{window}) setproperty!(pd,:swin,swin) end
function setStandard!(pd::sample,standard::Integer) setproperty!(pd,:standard,standard) end
export setStandard!

# set run attributes
function setSamples!(pd::run;samples::Vector{sample}) setproperty!(pd,:samples,samples) end
function setControl!(pd::run;ctrl::control) setproperty!(pd,:control,ctrl) end
function setBPar!(pd::run;bpar::Vector) setproperty!(pd,:bpar,bpar) end
function setSPar!(pd::run;spar::Vector) setproperty!(pd,:spar,spar) end
function setBCov!(pd::run;bcov::Matrix) setproperty!(pd,:bcov,bcov) end
function setSCov!(pd::run;scov::Matrix) setproperty!(pd,:scov,scov) end

# set key sample attributes in a run
function setBWin!(pd::run;i,bwin::Vector{window}) accessSample!(pd,i,setBWin!,bwin) end
function setSWin!(pd::run;i,swin::Vector{window}) accessSample!(pd,i,setSWin!,bwin) end
function setStandard!(pd::run;i,standard::Integer) accessSample!(pd,i,setStandard!,standard) end

# set control attributes
function setA!(ctrl::control,A::Vector{AbstractFloat}) setproperty!(pd,:A,A) end
function setB!(ctrl::control,B::Vector{AbstractFloat}) setproperty!(pd,:B,B) end
function setChannels!(ctrl::control,channels::Vector{String}) setproperty!(pd,:channels,channels) end

# set control attributes in a run
function setA!(pd::run,A::AbstractFloat) accessControl!(pd,:A,setA!,A) end
function setB!(pd::run,B::AbstractFloat) accessControl!(pd,:B,setB!,b) end
function setChannels!(pd::run,channels::Vector{String}) accessControl!(pd,:channels,setChannels!,channels) end

length(pd::run) = size(getSamples(pd),1)

function poolRunDat(pd::run,i=nothing)
    dats = getDat(pd,i=i)
    typeof(dats)
    reduce(vcat,dats)
end
