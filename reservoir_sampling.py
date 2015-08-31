import random

def random_element(stream):
    '''
    Return a random element from the iterable stream without using the len function.
    Only have one for loop over the stream.
    '''
    for k,x in enumerate(stream, start=1):
      if random.random() < 1.0 / k:
         chosen = x

    return chosen

def random_element_k(stream, k):
    '''
    Return a random element from the iterable stream without using the len function.
    Only have one for loop over the stream.
    '''
    k_samples = []
    
    for i, x in enumerate(stream):
        # Generate the reservoir
        if i <= k:
            k_samples.append(x)
        else:                  
            # Randomly replace elements in the reservoir
            # with a decreasing probability. by choosing 
            # an integer between 0 and i (index)               
            replace = random.randint(0, i-1)               
            if replace < k:                       
                k_samples[replace] = x

    return k_samples

print random_element([1,2,3,4,5,6,7,8])

print random_element_k([1,2,3,4,5,6,7,8], 2)