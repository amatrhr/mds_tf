# test 1: n points in n-1 dimensions
# idea: simulate a configuration of 10 points, hence 45 distances, 

import pytest
from numpy.testing import assert_approx_equal
from src.autograd_mds import *

def test_stress0():
    # set up example data  

    # 
    assert_approx_equal(actual=outstress, desired=0.0)