
require('nn')
require('rnn')


criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))

local seqLen = 5
local tgt = {}
mod = nn.LogSoftMax()
local a = torch.Tensor{{0.2, 0.3, 0.5}, {0.2, 0.3, 0.5}}
print(a)
a = a:resize(2, 3)

local input = {}

for i = 1, seqLen do
    table.insert(tgt, torch.Tensor{2,3})
    table.insert(input, a:clone())
end
for i = 1, #tgt do
    print(tgt[i])
end
print('---input--')
input[1][1][1] = 00
input[1][1][2] = 00
input[1][1][3] = 00
--input[1][1] = 0
input[1][2][3] = 0.5
for i = 1, #input do
    input[i] = mod:forward(input[i])
    print(input[i])

end
--input[5][2][3] = 0.6
print(input)
print(tgt)
--input = mod:forward(input)
print('input')
print(input)
local err
err = criterion:forward(input, tgt)
print(err)
