from numpy import minimum, maximum
def epsilon():
    return 1e-07
    
def min_max(values, clip_value_min, clip_value_max):
    n_min = minimum(values, clip_value_max)
    n_max = maximum(n_min, clip_value_min)
    return n_max
