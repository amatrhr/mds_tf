# test successive approximation
import pytest
import numpy as np
from numpy.testing import assert_approx_equal
from autograd import jacobian
from src.autograd_mds import get_pairwise_distances
## 

@pytest.fixture
def three_four_five_2d():
    three_four_five = np.array([[0.,0.], [0.,4.,], [3.,4.]])
    return three_four_five


def test_distance(three_four_five_2d):
    assert np.allclose(get_pairwise_distances(three_four_five_2d), np.array([16., 25., 9.]))


def test_dist_deriv_345(three_four_five_2d):
    """
    From Guttman:
    d(dist)/d(x_ha) = 2 *yija * (xia-xja) *(delta_hi - delta_hj)\
    = 2*abs((xia-xja))**0/(2*dij)*(xia-xja) *(delta_hi - delta_hj)
    =(xia-xja) *(delta_hi - delta_hj)/(dij))
    """
    gradient = jacobian(get_pairwise_distances)
    ## gradient should be a n(n-1)/2 x (n_obs x n_dim) tensor
    assert np.allclose(gradient(three_four_five_2d), )
    pass