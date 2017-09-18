
require('ffi')
require('torchx')
require('paths')

local M = {}

--[[
local function isvalid(opt, cachePath)
    local imagekk

end
]]

function M.create(opt, split)
    local cachePath = paths.concat(opt.gen, 've_dataset.t7')
    print('----------cachePath..'..cachePath)

    if not paths.filep(cachePath) then
        paths.mkdir('gen')

        local script = paths.dofile('ve-gen.lua')

        script.exec(opt, cachePath)
    end

    local videoInfo = torch.load(cachePath)

    local Dataset = require('datasets/ve_dataset')

    return Dataset(videoInfo, opt, split)
    
end

function M.createTrainTest(opt)
    local cachePath = paths.concat(opt.gen, 've_dataset.t7')
    print('----------cachePath..'..cachePath)

    if not paths.filep(cachePath) then
        paths.mkdir('gen')

        local script = paths.dofile('ve-gen.lua')

        script.exec(opt, cachePath)
    end

    local videoInfo = torch.load(cachePath)
    print('videoInfo')
    print(videoInfo)

    torch.manualSeed(0)

    local tr_idx = {}
    local ts_idx = {}
    for i = 1,#unique(videoInfo.train.videoClass) do
        --print('i  '..i)
        res = torch.find(videoInfo.train.videoClass, i)
        --print(res)
        local tr_num = torch.round( (#res) * 2 / 3)
        --[[print(res)
        print('#res '..#res)]]
        local perm = torch.randperm(#res)
        for jj = 1, tr_num do
            table.insert(tr_idx, res[perm[jj]])
        end
        for jj = tr_num + 1, #res do
            table.insert(ts_idx, res[perm[jj]])
        end
       --[[ print('perm')
        print(perm)
        print('tr_num '.. tr_num)
        print('tr_id')
        print(tr_idx)
        print('ts_idx')
        print(ts_idx) ]]

    end
    print('tr_idx ts_idx')
    print(#tr_idx)
    print(#ts_idx)
    --print(videoInfo.train.videoClass)
    
    local trVideoInfo = {}
    local tsVideoInfo = {}
    
    trVideoInfo.basedir = videoInfo.basedir
    trVideoInfo.classList = videoInfo.classList
    trVideoInfo.train = {}
    tsVideoInfo.basedir = videoInfo.basedir
    tsVideoInfo.classList = videoInfo.classList
    tsVideoInfo.test = {}
    
    trVideoInfo.train.videoPath = torch.CharTensor(#tr_idx, videoInfo.train.videoPath:size(2))
    trVideoInfo.train.videoClass = torch.LongTensor(#tr_idx)
    tsVideoInfo.test.videoPath = torch.CharTensor(#ts_idx, videoInfo.train.videoPath:size(2))
    tsVideoInfo.test.videoClass = torch.LongTensor(#ts_idx)

    print('trVideoInfo')
    print(trVideoInfo)
    print('tsVideoInfo')
    print(tsVideoInfo)
   
    for ii = 1, #tr_idx do
        trVideoInfo.train.videoPath[ii] = videoInfo.train.videoPath[tr_idx[ii]]:clone()
        trVideoInfo.train.videoClass[ii] = videoInfo.train.videoClass[tr_idx[ii]] 
--        print(string.format('videoPath %s , videoClass %d\n', ffi.string(trVideoInfo.train.videoPath[ii]:data()), trVideoInfo.train.videoClass[ii]))
    end
    
    for ii = 1, #ts_idx do
        tsVideoInfo.test.videoPath[ii] = videoInfo.train.videoPath[ts_idx[ii]]:clone()
        tsVideoInfo.test.videoClass[ii] = videoInfo.train.videoClass[ts_idx[ii]] 
    end
  
    print('videoInfo '..__LINE__())
    print(videoInfo)

    local Dataset = require('datasets/ve_dataset')

    return Dataset(trVideoInfo, opt, 'train'), Dataset(tsVideoInfo, opt, 'test')
    
end


--[[
return M
    local Dataset = require('datasets/ve_dataset')

    return Dataset(videoInfo, opt, 'train')
end]]



return M
