function PTerror(key)
    errormessages = Dict(
        "notStandard" => "The sample has not been marked as a standard",
        "unknownRefMat" => "Unknown reference material.",
        "unknownMethod" => "Unknown method.",
        "unknownInstrument" => "Unsupported instrument.",
        "missingNumDen" => "You must provide either a numerator or denominator, or both."
    )
    throw(error(errormessages[key]))
end
