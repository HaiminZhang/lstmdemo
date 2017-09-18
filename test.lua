
require('./util')
--print(util)
--myDebug()
print(__FILE__()..'..line..'..__LINE__())

a = torch.Tensor{1,2,3}
while true do
    local data = a
    data[1] = 100
    print(data)
    break
end
print(a)
function myprint() 
    print('in myprint()')
end


--return myprint
