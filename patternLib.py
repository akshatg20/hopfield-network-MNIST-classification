import numpy as np

# function to generate a random square pattern
def get_random_pattern(size):
    matrix = np.random.binomial(1, 0.5, size*size)
    matrix = matrix * 2 - 1  
    return matrix.reshape((size,size))
    
# function to generate a list of patterns
def get_patterns(size, num):
    patterns = list()
    for i in range(num):
        patterns.append(get_random_pattern(size))
    return patterns
    
# function to flip specific pixels of a pattern
def perturb_pattern(pattern, num_changes):
    indices = np.prod(pattern.shape)
    changes_indices = np.random.choice(indices, num_changes, replace=False)
    flat_pattern = pattern.flatten()
    flat_pattern[changes_indices] = - flat_pattern[changes_indices]
    return flat_pattern.reshape(pattern.shape)

# function to get difference between two patterns
def get_diff_pattern(pattern1, pattern2, diff_code = 0):
    if pattern1.shape != pattern2.shape:
        raise ValueError("patterns are not of equal shape")
    diff = np.multiply(pattern1, pattern2)
    diff_pattern = np.where(diff < 0, diff_code, pattern1)
    return diff_pattern
    
# function to calculate the overlap between two patterns
def compute_overlap(pattern1, pattern2):
    if pattern1.shape != pattern2.shape:
        raise ValueError("patterns are not of equal shape")
        
    dot_prod = np.dot(pattern1.flatten(), pattern2.flatten())
    return float(dot_prod) / (np.prod(pattern1.shape))
    
# function to calculate overlap between reference pattern and each pattern in given list
def compute_overlap_list(ref, pattern_list):
    overlap_list = np.zeros(len(pattern_list))
    for i in range(0, len(pattern_list)):
        overlap_list[i] = compute_overlap(ref, pattern_list[i])
    return overlap_list

# function to calculate the index of the first state from a list which matches the reference pattern
def time_taken_to_retrieve(ref, pattern_list):
    overlap_list = compute_overlap_list(ref, pattern_list)
    time = 0
    for overlap in overlap_list:
        if overlap == 1.0:
            return time
        else:
            time = time + 1
    # print("Unable to retrieve pattern")
    return time

# function to calculate whether a pattern was successfully retrieved by the network
def able_to_retrieve(ref, pattern_retrieved):
    if(compute_overlap(ref, pattern_retrieved) == 1.0):
        return 1
    else:
        return 0
    
