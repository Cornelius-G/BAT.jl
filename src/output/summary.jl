export Summary


struct Summary
    samples::DensitySampleVector
    chains::Any
end


function display_rich(io::IO, m::MIME, obj::Any)
    showable(m, obj) ?  show(io, m, obj) : print(io, obj)
end
