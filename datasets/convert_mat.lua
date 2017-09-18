
local paths = require('paths')
local matio = require('matio')

local srcDir = '/home/zhanghm/disk4t/user_gene_video/user_gene_10crop_MM17/data_txt'
local dstDir = '/home/zhanghm/disk4t/user_gene_video/resLSTM/data'
local dirs = paths.dir(srcDir)
--print(dirs)
local cnt = 0
for _, path in ipairs(dirs) do
    if path ~= '.' and path ~= '..' then
        print('path '..srcDir.. '/' ..path)
        local vids = paths.dir(srcDir .. '/' .. path)
        for _, tmpPath in ipairs(vids) do
            if tmpPath ~= '.' and tmpPath ~= '..' then 
                print('path '..srcDir..'/'..path..'/'..tmpPath)
                local f = io.open(srcDir..'/'..path..'/'..tmpPath)
                assert(f, tmpPath)
                --  read data here
                local nFea = f:read("*number")
                local dimFea = f:read("*number")
                print('nFea '..nFea)
                print('dimFea '..dimFea)
                local feaArrNorm = torch.FloatTensor(nFea, dimFea)
                print('feaSet size ')
                print(feaArrNorm:size())
                for ii = 1,nFea do
                    for jj = 1,dimFea do
                        --print('ii..'..ii)
                        local rawData = f:read("*num")
                        --print(' '..rawData)
                        feaArrNorm[ii][jj] = rawData
                    end
                    --[[print('path '..srcDir..'/'..path..'/'..tmpPath)
                    if ii == 2 then
                        os.exit(0) 
                    end]]
                end
                f:close()
                local saveDir = dstDir..'/'..path
                if not paths.dirp(saveDir) then
                    paths.mkdir(saveDir)
                end
                --print('saveDir..'..saveDir)
                fname = string.gmatch(tmpPath, "[^%.]+")()
                savePath = saveDir..'/'..fname..'.t7'
                print('tmpPath..'..savePath)
                local feaSet = {}
                feaSet.feaArrNorm = feaArrNorm
                torch.save(savePath, feaSet)
                cnt = cnt + 1
            end

        end
        --matio.load(rootDir .. '/' ..path)
        --matio.load(rootDir .. '/' ..path)
    end
end
print('# of videos '.. cnt)

