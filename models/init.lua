
require('rnn')
require('cunn')
require('cudnn')


local M = {}

function M.setup(opt)
    local model

    model = require('models/resModel')(opt)

    local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))
    --local criterion = nn.ClassNLLCriterion():type('torch.CudaTensor')

    return model, criterion
end


return M
