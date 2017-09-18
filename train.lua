
require('cutorch')
local optim = require('optim')

local M = {}

local Trainer = torch.class('resLSTM.Trainer', M)

function Trainer:__init(model, criterion, opt)
    self.model = model
    self.criterion = criterion
    self.optimState = {
        learningRate = opt.LR or 0.01,
        learningRateDecay = 0.0,
        momentum = 0.9,
        mesterov = true,
        dampening = 0,
        weightDecay = opt.weightDecay or 0,
    }
    
    self.opt = opt
   
    print('opt')
    print(self.opt)
    print('self.optimState')
    print(self.optimState)


    self.seqLen = opt.seqLen
    self.params, self.gradParams = model:getParameters()
end


local function targetToTable(target, seqLen)
    local batchSize = target:size(1)
    --[[print('target size '.. batchSize)
    print('target')
    print(target:float()) ]]
    target = target:float()
    fmt = {}
    for i = 1, seqLen do
        table.insert(fmt, target:float())
    end
    return fmt
end

function Trainer:learningRate(epoch)
    if epoch == 15 then
        self.optimState.learningRate = self.opt.LR * 0.5
    end
    if epoch == 30 then
        self.optimState.learningRate = self.opt.LR * 0.1
    end
end

function Trainer:computeScore(output, target)
--    print('outpu[1]')

    local mod = nn.CAddTable()
    output = mod:forward(output) 
    
    local _, res = output:float():topk(1, 2, true, true)
    --[[print('res')
    print(res)
    print('target')
    print(target)
    print('target:type()')
    print(target:type())
    print(res:type()) ]]
    local correct = res:eq(target:long())
    return correct:sum(), correct:size(1)
end

function Trainer:train(epoch, dataloader)

    print('--------dataloader')
    print(dataloader)

    self:learningRate(epoch)

    print('---epoch '..epoch..' , learningRate '.. self.optimState.learningRate)

    local function feval()
        print('self.criterion.output')
        print(self.criterion.output)
        print(string.format('File %s, line %d\n', __FILE__(), __LINE__()))

        return self.criterion.output, self.gradParams
    end

    self.model:training()
    for n, sample in dataloader:run() do
         print('epoch..'..epoch..' ------n '..n) 
         --[[print('optimState') 
         print(self.optimState) 
         os.exit(0) ]]

        if n <= 10000 then

            local tr_data = nn.utils.recursiveType(sample.input, 'torch.CudaTensor')
            local output = self.model:forward(tr_data)
            --[[print('#output')
            print(#output)
            print('self.target')

            print(sample.target) ]]
            fmt = targetToTable(sample.target, self.seqLen)
            --fmt = sample.target:clone()
            fmt = nn.utils.recursiveType(fmt, 'torch.CudaLongTensor')
            -- print('fmt')
            --print(fmt)
            --[[print('self.model.output')
            print(self.model.output[1])
            print(self.model.output[2])
            print('File '..__FILE__()..', .. line '..__LINE__()) ]]
            
            local loss = self.criterion:forward(self.model.output, fmt)
            
            self.model:zeroGradParameters()
            self.criterion:backward(self.model.output, fmt)
            self.model:backward(tr_data, self.criterion.gradInput)

            --[[print('---fmt')
            print(fmt[1]) 
            print('File '..__FILE__()..', .. line '..__LINE__())
            print(fmt[2]) 
            print('File '..__FILE__()..', .. line '..__LINE__())
            print('self.model')
            print(self.model)

            print('---loss '..loss) ]]

            optim.sgd(feval, self.params, self.optimState)
            --[[print('self.optimState')
            print(self.optimState) ]]
            --print(output.__typename)
            --print('--File '..__FILE__()..', __LINE__ '..__LINE__())
            if n == 200 then os.exit(0) end
            
            assert(self.params:storage() == self.model:parameters()[1]:storage())
            --print(sample)
            --print(sample.target)
            --print(sample.input[1].input[2][1])
            
            --print(sample.target)
        end
        --if n > 10 then os.exit(0) end
    end
end


function Trainer:test(epoch, dataloader)

    self.model:evaluate()
    print('file '..__FILE__()..', __LINE__ '..__LINE__())
    local nSample = 0
    local nTrue = 0
    for n, sample in dataloader:run() do
        
        --print('file '..__FILE__()..', __LINE__ '..__LINE__())
        local ts_data = nn.utils.recursiveType(sample.input, 'torch.CudaTensor')
        local output = self.model:forward(ts_data)
        --[[print('output')
        print(output) ]
        pre_data = output ]]
        local pre_data = {}
        for i = 1, #output do
            pre_data[i] = output[i]:float()

        end
        --nSmpl = nSmpl + pre_data[1]:size(1)
        --print('output line '..__LINE__())
        --print(pre_data)
        --nn.utils.recursiveType(output, 'torch.FloatTensor')
        --print('pre_data  n = '..n)
        local true_pred, batchSize = self:computeScore(pre_data, sample.target)
        --print('batchSize')
        --print(batchSize)
        nTrue = nTrue + true_pred
        nSample = nSample + batchSize
    end
    print('# of testing samples '..nSample..' , true prediction '..nTrue..' , accuracy '..nTrue/nSample)

end


return M.Trainer
