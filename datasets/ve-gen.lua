
local ffi = require('ffi')
require('paths')

local M = {}


local function findClasses(dir)
    local dirs = paths.dir(dir)
    print('--------dirs--')
    print(dirs)
    table.sort(dirs)
    print('--------dirs--')
    print(dirs)
    
       local classList = {}
   local classToIdx = {}
   local cnt = 0
   for _ ,class in ipairs(dirs) do
      if not classToIdx[class] and class ~= '.' and class ~= '..' and class ~= '.DS_Store' then
         table.insert(classList, class)
         classToIdx[class] = #classList

         --[[print('classList----')
         print(classList)
         print('classToIdx----')
         print(classToIdx)
         cnt = cnt + 1
         if cnt == 10 then
            os.exit(0)
         end ]]
      end
   end

   print('----classList')
   print(classList)

   print('-------classToIdx---')
   print(classToIdx)
   -- assert(#classList == 1000, 'expected 1000 ImageNet classes')
   return classList, classToIdx
    
end


local function findVideoFeaData(dir, classToIdx)

    local vidPath = torch.CharTensor()
    local vidClass = torch.LongTensor()

    local findOptions = 'find -L '..dir..' -iname "*.t7"'
    print('findOptions')
    print(findOptions)


    local f = io.popen(findOptions)

    local maxLength = -1  --maxLength per file path
    local videoPaths = {}
    local videoClasses = {}

    cnt = 1
    while true do
        local line = f:read('*line')
        if not line then 
            break 
        end

        print('cnt ')
        print(cnt)

        cnt = cnt + 1
        --print(line)
        local className = paths.basename(paths.dirname(line))
        --print('className..'..className)
        local filename = paths.basename(line)
        local path = className .. '/' .. filename
        print('path  '..path)
        local classId = classToIdx[className]
        assert(classId, 'class not found: '..className)

        table.insert(videoPaths, path)
        table.insert(videoClasses, classId)

        maxLength = math.max(maxLength, #path + 1)
    end

    f:close() 
    print('# of videos '.. cnt)


    local nVideos = #videoPaths
    local videoPath = torch.CharTensor(nVideos, maxLength):zero()

    for i, path in ipairs(videoPaths) do
        ffi.copy(videoPath[i]:data(), path)
    end

    local videoClass = torch.LongTensor(videoClasses)

    return videoPath, videoClass
end

function M.exec(opt, cacheFile)
   local vidPath = torch.CharTensor()
   local vidClass = torch.LongTensor()

   local dataDir = '/home/zhanghm/disk4t/user_gene_video/resLSTM/data'
   --local valDir
   assert(paths.dirp(dataDir), 'data directory not found: '.. dataDir)
   local classList, classToIdx = findClasses(dataDir)

   --[[
   print('-----------classList')
   print(classList)
   print('-----------classToIdx')
   print(classToIdx)]]

   local videoPath, videoClass = findVideoFeaData(dataDir, classToIdx)
   print('videoPath, videoClass')
   print(videoPath:size())
   print(videoClass:size())
   --[[print('videoPath')
   print(videoPath) 
   print('--videoClass')
   print(videoClass) ]]
   
   local info = {
      basedir = opt.dataDir,
      classList = classList,
      train = {
         videoPath = videoPath,
         videoClass = videoClass,
      },
   }

   print(" | saving list of videos to "..cacheFile)
   torch.save(cacheFile, info)
   return info


end

return M
