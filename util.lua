
--local M = {}

function __FILE__() 
    return debug.getinfo(2, 'S').source
end

function __LINE__()
    --return debug.getinfo(2, '1').currentline
    return debug.getinfo(2, 'l').currentline
end

--[[
unction myDebug()
    return "FILE of "__FILE__()..', LINE at '..__LINE__
end
]]

--return M
function unique(input)
    local b = {}
    for i = 1,input:numel() do
        b[input[i]] = true
    end
    local out = {}
    for i in pairs(b) do
        table.insert(out, i)
    end
    return out
end
