
require('rnn')


local function createModel(opt)
    local model = nn.Sequential()
    local inputSize = opt.inputSize
    local timeStep = opt.timeStep
    local hiddenSizeLSTM = opt.hiddenSizeLSTM
    local nClasses = opt.nClasses

    
    model:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSizeLSTM):maskZero(1)))
    --model:add(nn.Sequencer(nn.resLSTM(inputSize, hiddenSizeLSTM):maskZero(1)))
    model:add(nn.Sequencer(nn.MaskZero(nn.Linear(hiddenSizeLSTM, nClasses), 1)))
    model:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))


--[[    model:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSizeLSTM):maskZero(1)))
    --model:add(nn.Sequencer(nn.resLSTM(inputSize, hiddenSizeLSTM):maskZero(1)))
    model:add(nn.Sequencer(nn.MaskZero(nn.Linear(hiddenSizeLSTM, nClasses), 1)))
    model:add(nn.SelectTable(-1))
    model:add(nn.LogSoftMax()) ]]

    return model


end


return createModel
