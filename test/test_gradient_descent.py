# test successive approximation
import pytest
import numpy as np
from numpy.testing import assert_approx_equal
from autograd import jacobian
from src.autograd_mds import get_pairwise_distances
## 

def ugly_jacobian(arr, pwds):
    ## keep track of difference indices 
    ijs = list(zip(np.triu_indices(arr.shape[0],k=1)[0], np.triu_indices(arr.shape[0],k=1)[1]))
    
    ## tensor out
    out_array = np.zeros((len(pwds), *arr.shape))
    
    #start with triple loop-- Structure of the Jacobian is (number of distances, number of observations, dimension of config)
    for ij_idx, dij in enumerate(pwds):
        ##
        i, j = ijs[ij_idx]
        for h in range(arr.shape[0]):
            delta_hi = float(h == i)
            delta_hj = float(h == j)
            
            for a in range(arr.shape[1]):
                ## flip sign for other
                ## how to apply kronecker delta here?
                out_array[ij_idx, h, a] = ((arr[i, a] - arr[j, a])*(delta_hi - delta_hj))/dij if dij != 0. else 0.
                
    return out_array

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
    local_pwds = get_pairwise_distances(three_four_five_2d)
    closed_form_jacobian = ugly_jacobian(three_four_five_2d, local_pwds)
    ## gradient should be a n(n-1)/2 x (n_obs x n_dim) tensor
    assert np.allclose(gradient(three_four_five_2d), closed_form_jacobian)
    return