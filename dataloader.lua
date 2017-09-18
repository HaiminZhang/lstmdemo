
local datasets = require('datasets/init')

local Threads = require('threads')
Threads.serialization('threads.sharedserialize')


local M = {}

local DataLoader = torch.class('resLSTM.DataLoader', M)

function DataLoader.create(opt)
     local loaders = {}
     local tr_ds, ts_ds = datasets.createTrainTest(opt)
     loaders[1] = M.DataLoader(tr_ds, opt, 'train')
     loaders[2] = M.DataLoader(te_ds, opt, 'test')

    print('loaders[1]') 
    print(loaders[1]) 
    print('loaders[2]') 
    print(loaders[2]) 
    os.exit(0)
--     M.DataLoader(dataset, opt, split)
     --return table.unpack(loders)
     return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
    local manualSeed = opt.manualSeed
    local function init()
        require('datasets/'..opt.dataset)
    end
    
    local function main(idx)
        assert(opt.manualSeed == 0, 'opt.manualSeed '..opt.manualSeed)
        torch.setnumthreads(1)
        _G.dataset = dataset

        return dataset:size()
    end

    local threads, sizes = Threads(opt.nThreads, init, main)
    self.threads = threads
    print('sizes')
    --print(sizes)
    self.__size = sizes[1][1]
    self.batchSize = opt.batchSize
    self.seqLen = opt.seqLen

    local function getCPUType()
        return 'FloatTensor'
    end
    self.cpuType = getCPUType()
    
end

local function videoSeqPadding(sample, seqLen)
    --print('videoSeqPadding')
    local nFea = sample:size()[1]
    local dimFea = sample:size()[2]
    --print('nFea '..nFea)
    --print('dimFea '..dimFea)
    local data = torch.FloatTensor(seqLen, dimFea):zero()
    --print('data:size() ')
    --print(data:size())
    if nFea <= seqLen then
        local start = seqLen - nFea + 1
        for i = start,seqLen do
            --print('i '..i)
            data[i] = sample[i - start + 1]
        end
        return data
    end

    local start = nFea - seqLen + 1
    return sample:narrow(1, start, seqLen)
   
end

local function seqPacking(batch, seqLen)
    --print('# of samples in a batch '..#batch)
    local dimFea = batch[1]:size()[2]
    local batchSize = #batch--batch[1]:size()[1]
    --[[print('dimFea')
    print(dimFea)
    print('batchSize')
    print(batchSize) ]]
    --os.exit(0)]]
    local data = {}
    for i = 1, seqLen do
        local tmp = torch.FloatTensor(batchSize, dimFea)
        for j = 1, batchSize do
            tmp[j] = batch[j][i]:clone()
        end

        table.insert(data, tmp)

    end
    --[[
    print('data')
    print(data[1][1][2048])
    print(data[2][1][2048])
    print(data[1][2][2048])
    print(data[2][2][2048])
    ]]

    return data
end

function DataLoader:run()
    print('-------in DataLoader:run()')
    local threads = self.threads
    local size, batchSize = self.__size, self.batchSize
    print('---size '..size)
    local perm = torch.randperm(size)

    local idx, sample = 1, nil
    local function enqueue()
        while idx <= size and threads:acceptsjob() do
            --print('---idx  '..idx)
            local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))

            threads:addjob(
                function (indices, cpuType, seqLen) 
                    local sz = indices:size(1)
                    local batch = {}, videoSize
                    local target = torch.IntTensor(sz)

                    for i, idx in ipairs(indices:totable()) do
                        local sample = _G.dataset:get(idx)
                        
                        --[[print('----self.batchSize ')
                        print(seqLen)
                        print('sample')
                        print({sample.input}) ]]

                        smp = videoSeqPadding(sample.input, seqLen)
                        --torch.save('sample.t7', sample)
                        --os.exit(0)
                        
                        --local input = _G.preprocess(sample.input)
                        --[[if not batch then
                            batch = torch[cpuType](sz)
                        end ]]
                        batch[i] = smp
                        target[i] = sample.target
                    end
                    local seqPacked = seqPacking(batch, seqLen)
                   --[[ print('seqPacked')
                    print(seqPacked)
                    os.exit(0) ]]
                    collectgarbage()
                    return {
                        input = seqPacked,
                        target = target,
                    }
                end,
                function (_sample_) --returned by the function above
                    sample = _sample_
                end,
                indices,
                self.cpuType,
                self.seqLen
            )
            idx = idx + batchSize
        end
    end

    local n = 0
    local function loop()
        enqueue()
        if not threads:hasjob() then
            return nil
        end
        threads:dojob()
        if threads:haserror() then
            threads:synchronize()
        end
        enqueue()
        n = n + 1
        return n, sample
    end

    return loop
end

return M.DataLoader
