# Get location of project root
import os, inspect
try:
    frame = inspect.currentframe()
    DIR_PYROSS = os.path.dirname(inspect.getfile(frame))
finally:
    # Always manually delete frame
    # https://docs.python.org/2/library/inspect.html#the-interpreter-stack
    del(frame)


import pyross.contactMatrix
import pyross.control
import pyross.deterministic
import pyross.hybrid
import pyross.inference
import pyross.stochastic
import pyross.forecast
import pyross.utils
