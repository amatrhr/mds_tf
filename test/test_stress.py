# test 1: n points in n-1 dimensions
# idea: simulate a configuration of 10 points, hence 45 distances, 
# then fit to a configuration in 9-d
import pytest
from numpy.testing import assert_approx_equal
from src.autograd_mds import *

def test_stress0():
    # set up example data  
    outstress =0.0
    # 
    assert_approx_equal(actual=outstress, desired=0.0)