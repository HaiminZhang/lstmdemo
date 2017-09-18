
local paths = require('paths')
local ffi = require('ffi')
local matio = require('matio')


local M = {}

local VEDataset = torch.class('resLSTM.VEDataset', M)

function VEDataset:__init(vidInfo, opt, split)
    self.videoInfo = vidInfo[split]
    print('videoInfo  '..__LINE__())
    print(self.videoInfo)
    assert(self.videoInfo, 'self.videoInfo nil  '..split)
    self.opt = opt
    self.split = split
    self.dir = paths.concat(opt.dataDir)

end

function VEDataset:size()
    --print('self.videoInfo')
    --print(self.videoInfo)
    return self.videoInfo.videoClass:size(1)
end

function VEDataset:get(i)
    local path = ffi.string(self.videoInfo.videoPath[i]:data())
    --[[print('----i--')
    print(i)
    print('self.dir ..'..self.dir.. ' path '..path)
    ]]
    local videoFea = self:_loadVideFea(paths.concat(self.dir, path))
    
    videoFea = videoFea * 10

    local class = self.videoInfo.videoClass[i]
    return {
        input = videoFea,
        target = class,
    }
end

function VEDataset:_loadVideFea(path)
    --print('path '..path)
    --
    local feaSet = torch.load(path)
    return feaSet.feaArrNorm
end
return M.VEDataset
