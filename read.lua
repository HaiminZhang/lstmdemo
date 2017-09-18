
local f = io.open('data.dat', "r")
assert(f)

while true do
    local d = f:read("*number")
    if not d then 
        break
    end

    print('d '..d)
end
