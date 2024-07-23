# include("/home/pvermees/git/Plasmatrace/test/emacs.jl")

using Test, CSV
import Plots

function loadtest(verbose=false)
    myrun = load("data/Lu-Hf";instrument="Agilent")
    if verbose summarise(myrun;verbose=true,n=5) end
    return myrun
end

function plottest()
    myrun = loadtest()
    p = plot(myrun[1],["Hf176 -> 258","Hf178 -> 260"])
    @test display(p) != NaN
    p = plot(myrun[1],["Hf176 -> 258","Hf178 -> 260"], den="Hf178 -> 260")
    @test display(p) != NaN
end

function windowtest()
    myrun = loadtest()
    i = 2
    setSwin!(myrun[i],[(70,90),(100,140)])
    p = plot(myrun[i],["Hf176 -> 258","Hf178 -> 260"])
    @test display(p) != NaN
end

function blanktest()
    myrun = loadtest()
    blk = fitBlanks(myrun;nblank=2)
    return myrun, blk
end

function standardtest(verbose=false)
    myrun, blk = blanktest()
    standards = Dict("BP" => "BP")
    setGroup!(myrun,standards)
    anchors = getAnchors("Lu-Hf",standards)
    if verbose
        println(anchors)
        summarise(myrun;verbose=true,n=5)
    end
end

function predictest()
    myrun, blk = blanktest()
    method = "Lu-Hf"
    channels = Dict("d" => "Hf178 -> 260",
                    "D" => "Hf176 -> 258",
                    "P" => "Lu175 -> 175")
    glass = Dict("NIST612" => "NIST612p")
    setGroup!(myrun,glass)
    standards = Dict("BP" => "BP")
    setGroup!(myrun,standards)
    fit = Pars([4.2670587703673934], [0.0, 0.05197296298083967], 0.3838697441780825)
    samp = myrun[4]
    if samp.group == "sample"
        println("Not a standard")
    else
        pred = predict(samp,method,fit,blk,channels,standards,glass)
        p = plot(samp,method,channels,blk,fit,standards,glass;transformation="log")
        @test display(p) != NaN
    end
    return pred
end

function fractionationtest(all=true)
    myrun, blk = blanktest()
    method = "Lu-Hf"
    channels = Dict("d" => "Hf178 -> 260",
                    "D" => "Hf176 -> 258",
                    "P" => "Lu175 -> 175")
    glass = Dict("NIST612" => "NIST612p")
    setGroup!(myrun,glass)
    standards = Dict("BP" => "BP")
    setGroup!(myrun,standards)
    if all
        print("two separate steps: ")
        mf = fractionation(myrun,method,blk,channels,glass)
        fit = fractionation(myrun,method,blk,channels,standards,mf;
                            ndrift=1,ndown=1)
        println(fit)
        print("no glass: ")
        fit = fractionation(myrun,method,blk,channels,standards,nothing;
                            ndrift=1,ndown=1)
        println(fit)
        print("two joint steps: ")
    end
    fit = fractionation(myrun,"Lu-Hf",blk,channels,standards,glass;ndrift=1,ndown=1)
    if (all)
        println(fit)
        return myrun, blk, fit, channels, standards, glass
    else
        Ganchors = getAnchors(method,glass,true)
        Sanchors = getAnchors(method,standards,false)
        anchors = merge(Sanchors,Ganchors)
        return myrun, blk, fit, channels, standards, glass, anchors
    end
end

function histest()
    myrun, blk, fit, channels, standards, glass, anchors = fractionationtest(false)
    pooled = pool(myrun;signal=true,group="BP")
    anchor = anchors["BP"]
    pred = predict(pooled,fit,blk,channels,anchor)
    misfit = @. pooled[:,channels["d"]] - pred[:,"d"]
    p = Plots.histogram(misfit;legend=false)
    @test display(p) != NaN
end

function averatest()
    myrun, blk, fit, channels, standards, glass, anchors = fractionationtest(false)
    t, T, P, D, d = atomic(myrun[1],channels,fit,blk)
    ratios = averat(myrun,channels,fit,blk)
    println(first(ratios,5))
    return ratios
end

function processtest()
    myrun = load("data/Lu-Hf",instrument="Agilent")
    method = "Lu-Hf"
    channels = Dict("d"=>"Hf178 -> 260",
                    "D"=>"Hf176 -> 258",
                    "P"=>"Lu175 -> 175")
    standards = Dict("Hogsbo" => "hogsbo")
    glass = Dict("NIST612" => "NIST612p")
    blk, fit = process!(myrun,method,channels,standards,glass,
                        nblank=2,ndrift=2,ndown=2)
    p = plot(myrun[2],method,channels,blk,fit,standards,glass,
             den="Hf176 -> 258",transformation="log")
    @test display(p) != NaN
end

function PAtest(verbose=false)
    myrun = load("data/Lu-Hf",instrument="Agilent")
    method = "Lu-Hf"
    channels = Dict("d"=>"Hf178 -> 260",
                    "D"=>"Hf176 -> 258",
                    "P"=>"Lu175 -> 175")
    standards = Dict("Hogsbo" => "hogsbo")
    glass = Dict("NIST612" => "NIST612p")
    cutoff = 1e7
    blk, fit = process!(myrun,method,channels,standards,glass;
                        PAcutoff=cutoff,nblank=2,ndrift=1,ndown=1)
    ratios = averat(myrun,channels,fit,blk)
    if verbose println(first(ratios,5)) end
    return ratios
end

function exporttest()
    ratios = PAtest()
    selection = subset(ratios,"BP") # "hogsbo"
    CSV.write("BP.csv",selection)
    export2IsoplotR(selection,"Lu-Hf",fname="BP.json")
end

function RbSrtest()
    myrun = load("data/Rb-Sr",instrument="Agilent")
    method = "Rb-Sr"
    channels = Dict("d"=>"Sr86 -> 102",
                    "D"=>"Sr87 -> 103",
                    "P"=>"Rb85 -> 85")
    standards = Dict("MDC" => "MDC -")
    setGroup!(myrun,standards)
    blank = fitBlanks(myrun,nblank=2)
    fit = fractionation(myrun,method,blank,channels,standards,1.0;
                        ndrift=1,verbose=false)
    anchors = getAnchors(method,standards)
    p = plot(myrun[2],channels,blank,fit,anchors,transformation="log",den="Sr86 -> 102")
    export2IsoplotR(myrun,method,channels,fit,blank,prefix="Entire",fname="Entire.json")
    @test display(p) != NaN
end

function UPbtest()
    myrun = load("data/U-Pb",instrument="Agilent",head2name=false)
    method = "U-Pb"
    standards = Dict("Plesovice" => "STDCZ",
                     "91500" => "91500")
    glass = Dict("NIST610" => "610",
                 "NIST612" => "612")
    channels = Dict("d"=>"Pb207","D"=>"Pb206","P"=>"U238")
    blank, pars = process!(myrun,"U-Pb",channels,standards,glass,
                           nblank=2,ndrift=1,ndown=1)
    export2IsoplotR(myrun,method,channels,pars,blank,fname="UPb.json")
    p = plot(myrun[1],method,channels,blank,pars,standards,glass,transformation="log")
    @test display(p) != NaN
end

function iCaptest(verbose=true)
    myrun = load("data/iCap",instrument="ThermoFisher")
    if verbose summarise(myrun;verbose=true,n=5) end
end

function carbonatetest(verbose=false)
    method = "U-Pb"
    myrun = load("data/carbonate",instrument="Agilent")
    standards = Dict("WC1"=>"WC1")
    glass = Dict("NIST612"=>"NIST612")
    channels = Dict("d"=>"Pb207","D"=>"Pb206","P"=>"U238")
    blk, fit = process!(myrun,method,channels,standards,glass,
                        nblank=2,ndrift=1,ndown=1,verbose=verbose)
    export2IsoplotR(myrun,method,channels,fit,blk,prefix="Duff",fname="Duff.json")
    p = plot(myrun[3],method,channels,blk,fit,standards,glass,
             transformation=nothing,num=["Pb207"],den="Pb206",ylim=[-0.02,0.3])
    @test display(p) != NaN
end

function timestamptest(verbose=true)
    myrun = load("data/timestamp/MSdata.csv","data/timestamp/timestamp.csv";
                 instrument="Agilent")
    if verbose summarise(myrun;verbose=true,n=5) end
    return myrun
end

function TUItest()
    PT("logs/test.log")
end

Plots.closeall()

#=@testset "load" begin loadtest(true) end
@testset "plot raw data" begin plottest() end
@testset "set selection window" begin windowtest() end
@testset "set method and blanks" begin blanktest() end
@testset "assign standards" begin standardtest(true) end
@testset "predict" begin predictest() end
@testset "fit fractionation" begin fractionationtest() end
@testset "hist" begin histest() end
@testset "average sample ratios" begin averatest() end
@testset "process run" begin processtest() end
@testset "PA test" begin PAtest(true) end
@testset "export" begin exporttest() end
@testset "Rb-Sr" begin RbSrtest() end
@testset "U-Pb" begin UPbtest() end
@testset "iCap test" begin iCaptest() end
@testset "carbonate test" begin carbonatetest() end=#
@testset "timestamp test" begin timestamptest() end
#@testset "TUI test" begin TUItest() end
