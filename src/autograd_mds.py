import autograd.numpy as np
from autograd import grad
from loguru import logger 
import matplotlib.pyplot as plt
from copy import deepcopy
from data_cleaning import generate_starting_configuration
from autograd import elementwise_grad  

## TODO move to matrix_utils.py 
## MAYBE rewrite in pure tensorflow
### need config_dim 
### -> then apply np.outer on each to get Aij
def self_outer(basis_vector:np.array)->np.array:
    return np.outer(basis_vector, basis_vector)
    

def pairwise_basis_diffs(max_index:int, max_rows:int)->np.array:
  
    assert max_index > 1 ## ignore first basis vector  
    eye_ref = np.eye(N=max_rows, M = max_index)
    
    # for index i and all other indices j less than it , build a 2-d array of (e_is - e_js)s
    # by subtracting highest indexed basis vector from all other basis vectors 
    ei_minus_ejs = eye_ref[:,:-1] - eye_ref[:,-1].reshape(max_rows,1)
    # find outer-product with self for each difference of basis columns (these are the A_ij s) 
    # where i is fixed at index 
    outer_list = np.apply_along_axis(func1d=self_outer, axis=0, arr=ei_minus_ejs)
    return outer_list 
### then apply func1d on Aijs 
def pairwise_dist_block_chunk(outer_distance:np.array, configuration:np.array):
    # Compute tr X'A_ij X 
    return np.trace(configuration.T @ outer_distance @ configuration)
    
## then, use listcomp to get chunk of pairwise distances and vstack the pairwise distances 
def pairwise_dist_within_block(outer_list:np.array, configuration:np.array)->np.array:
    pairwise_dist_in_block = [pairwise_dist_block_chunk(outer_list[:,:,k], configuration) for k in range(outer_list.shape[2]) ]
    return np.array(pairwise_dist_in_block)

def pairwise_dists_of_block(index:int, configuration:np.array)->np.array:
    outerL = pairwise_basis_diffs(index, configuration.shape[0])
    return pairwise_dist_within_block(outerL, configuration)

## wrap 
def get_pairwise_distances(configuration):
    ## Wrapper for pairwise distance function (if prep needed)
    return np.hstack([pairwise_dists_of_block(i, configuration) for  i in range(2, configuration.shape[0]+1)])

def stress(y_true, y_pred):
    Sstar = np.sum((y_true - y_pred)**2,axis=-1)
    print(f"Sstar: {Sstar}")
    Tstar = np.sum(y_true**2,axis=-1)
    print(f"Tstar: {Tstar}")
    S = np.sqrt(np.divide(Sstar,Tstar))
    return S  # Note the `axis=-1`

def angle_factor(old_grad, new_grad):
    old_grad_line = np.ravel(old_grad)
    new_grad_line = np.ravel(new_grad)
    dotprod = np.dot(old_grad_line, new_grad_line)
    denom = np.sqrt(np.linalg.norm(old_grad_line)*np.linalg.norm(new_grad_line))
    coss = dotprod/denom
    af = np.power(4.0, coss)**3.0 
    return af

# implement monotone regression
def check_satisfied(partition, index, direction):
    if index >= len(partition)-1 or index <= 0:
        return True
    if direction == 'up':
        next_index = index + 1 
        try: 
            assert next_index < len(partition)
        except:
            logger.info("Warning: Out of range")
        if np.mean(partition[index]) < np.mean(partition[next_index]):
            return True
    elif direction == 'down':
        prev_index = index - 1
        assert prev_index > -1
        if np.mean(partition[index]) > np.mean(partition[prev_index]):
            return True
    return False

def merge_blocks(old_partition, index, direction):
    partition = deepcopy(old_partition)
    if direction == 'up':
        assert index + 1 < len(partition)
        partition[index:index+2] = [partition[index] + partition[index+1]]
        logger.info("merged up")
    elif direction == 'down':
        assert index - 1 > -1
        partition[index-1:index+1] = [partition[index] + partition[index-1]]
        logger.info("merged down")
    else:
        raise Exception('no direction')
#     logger.info(f"returning partition: {partition}")
    return partition

def eval_direction_expand(partition, active_index, direction):
    next_direction = "up" if direction == "down" else "down"
    indx_chg = 0
    satisfied = check_satisfied(partition, active_index, direction)
    logger.info(f"currently {satisfied}ly {direction}-satisfied")
    new_partition = deepcopy(partition)
    if not satisfied:
        indx_chg = -1 if direction == "down" else 0 
        logger.info(f"merging {direction} at index {active_index} of partition {new_partition}")
        new_partition = merge_blocks(new_partition, active_index, direction)
    
    return new_partition, next_direction, satisfied, indx_chg

def stabilize_block(new_partition, active_index):
    up_satisfied = False if active_index > 0 else True
    down_satisfied = False if active_index < (len(new_partition)-1) else True
    direction = "up"
    directional_holder = {"up": up_satisfied, "down":down_satisfied}
    iterat = 0
    indx_chg = 0 
    while (not directional_holder["up"]) or (not directional_holder["down"]):
        # check up-satisfied 
        logger.info(directional_holder)
        old_direction = direction
        old_partition = new_partition
        new_partition, direction, is_satisfied, indx_chg = eval_direction_expand(old_partition, active_index, old_direction)
        directional_holder[old_direction] = is_satisfied
        if not np.array_equal(old_partition, new_partition):
            logger.info("Partition Change!")
            directional_holder[old_direction] = False
        active_index += indx_chg
        logger.info (f"at end of iteration {iterat}, {directional_holder} on partition{new_partition}; active_index {active_index}")
        iterat += 1
    return new_partition

class MonoReg:
    def __init__(self, pred_dis, true_dis, verbose=True):
#         breakpoint()
        self.pred_dis = np.array(pred_dis)
        self.true_dis = np.array(true_dis)
        assert len(self.pred_dis) == len(self.true_dis)
        self.sort_dict = np.array([range(len(self.true_dis)), np.argsort(self.true_dis)])
        self.unsort = [int(x) for x in np.lexsort(self.sort_dict)]
        self.order = np.argsort(self.true_dis)
        self.true_dis = self.true_dis[self.order]
        self.pred_dis = self.pred_dis[self.order]
        self.partition = [[x] for x in self.pred_dis]
        if not verbose:
            logger.add (sys.stdout, format=" {time} {level} {message}", filter="my_module", level="DEBUG")
        
    def apply_mean(self):
        self.pred_dis_out = [[np.mean(y) for z in y] for y in self.partition]
        self.pred_dis_out = [*it.chain.from_iterable([[np.mean(y) for z in y] for y in self.pred_dis_out])]
#         breakpoint()
        plt.scatter(self.pred_dis_out, self.true_dis)
        plt.show()
        plt.close()
        logger.info("PRED_DIS_OUT:")
        logger.info(self.pred_dis_out)
        self.first_diffs = [self.pred_dis_out[1:][x] - self.pred_dis_out[:-1][x] for x in range(len(self.pred_dis_out) - 1)]
        return
        
         
    def run_monoreg(self):
        # loop over active_blocks -- now, 
        for block  in range(len(self.partition)):
            new_partition = deepcopy(self.partition)
            self.partition = stabilize_block(new_partition, block)
            if block == len(self.partition) - 1:
                break
            # 
        self.apply_mean()
        logger.info(f"FIRST DIFFS:\n{self.first_diffs}" )
        while any([x < 0 for x in self.first_diffs]):
            for block  in range(len(self.partition)):
                new_partition = deepcopy(self.partition)
                self.partition = stabilize_block(new_partition, block)
                if block == len(self.partition) - 1:
                    break
            # 
            self.apply_mean()
            
        self.pred_dis_out = [self.pred_dis_out[x] for x in self.unsort]
        plt.scatter(self.pred_dis_out, self.true_dis[self.unsort])
        plt.show()
        plt.close()
        logger.info(f"FIRST DIFFS:\n{self.first_diffs}" )
        return self.pred_dis_out
   
def my_mds_training_loop(dissimilarities, n_init, eps):
    combi_config = np.zeros_like(generate_starting_configuration(dissimilarities, n_samples) )
    for test in range(n_init):
        start_config = generate_starting_configuration(dissimilarities, n_samples) 
        def get_stress_for_real_true(config):
            return stress(dissimilarities, get_pairwise_distances(config))

        gradiente = elementwise_grad(get_stress_for_real_true)
        # training loop
        old_stress = 1e6
        diff = 1e3 
        alpha = 0.8
        loop_config = deepcopy(start_config)
        start_grad = np.ravel(start_config)
        start_grad /= np.linalg.norm(start_grad)
        old_stress = 1
        iteration = 0
        stress_vec = []
        while diff > eps:
            pwds = get_pairwise_distances(loop_config)

            # monotone regression
            mpwds = MonoReg(pred_dis = pwds, true_dis = dissimilarities).run_monoreg()
            # end monotone regression
            iter_stress = stress(dissimilarities, mpwds)
            print(f"stress: {iter_stress}")
            stress_grad = gradiente(loop_config)
            grad_mag = np.linalg.norm(stress_grad)
            af = angle_factor(start_grad,stress_grad)
            good_luck = min(1, iter_stress/old_stress)
            five_iter = 1 if iteration < 5 else min(1, iter_stress/stress_vec[iteration-5])
            alpha = af*good_luck*five_iter
            loop_config += -1.*alpha*stress_grad*(1./grad_mag)
            diff = abs(old_stress - iter_stress)
            old_stress = iter_stress
            stress_vec.append(iter_stress)
            iteration += 1
            start_grad = stress_grad
        combi_config += 0.1*np.nan_to_num(loop_config)
    return combi_config