from __future__ import division
import cDpm
from numpy import *
from matplotlib.pyplot import *

alpha=linspace(1,10,1000)
llh = cDpm.calc_alpha_llh(alpha, 3.0, 1.0, 15, 50)
plot(alpha, llh)
show()