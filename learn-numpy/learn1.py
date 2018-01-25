from numpy import *

randMat = mat(random.rand(4, 4))
print(randMat)
print(randMat.I)
print(randMat * randMat.I)
eye(4)