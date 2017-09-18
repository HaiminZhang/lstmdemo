
version = 0.1

require('./util.lua')
require('torch')
require('optim')
require('nn')
require('cutorch')

local DataLoader = require('dataloader')
local Trainer = require('train')
local models = require('models/init')


torch.setdefaulttensortype('torch.FloatTensor')


local opt = {}
opt.dataset = 've_dataset'
opt.dataDir = '/home/zhanghm/disk4t/user_gene_video/resLSTM/data'
opt.inputSize = 2048
opt.hiddenSizeLSTM = 1024
opt.seqLen = 256
opt.epoch = 1024
opt.gen = 'gen'
opt.manualSeed = 0
opt.nThreads = 16
opt.batchSize = 64
opt.nEpochs = 200
opt.nClasses = 8
opt.LR = 0.001

resModel = require('models/resModel')

local trLoader = DataLoader.create(opt)
print('trLoader')
print(trLoader)
--print(trLoader:run())

--trLoader:get(1)
--
local model, criterion = models.setup(opt)
model = model:cuda()
criterion = criterion:cuda()

print('model')
print(model)


local trainer = Trainer(model, criterion, opt)

local startEpoch = 1
for epoch = startEpoch, opt.nEpochs do
    trainer:train(epoch, trLoader)
end
print('file '..__FILE__()..' LINE '..__LINE__())
