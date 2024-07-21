module RecursiveArrayToolsExt

import RecursiveArrayTools: recursivecopy, recursivecopy!
import PSDMatrices: PSDMatrix

recursivecopy(M::PSDMatrix) = PSDMatrix(recursivecopy(M.R))
recursivecopy!(dst::PSDMatrix, src::PSDMatrix) = (recursivecopy!(dst.R, src.R); dst)

end
