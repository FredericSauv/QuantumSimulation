#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:58:58 2019

@author: fred
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:58:58 2019

@author: fred
"""
import scipy.stats
import math
def prob(p1, p2, n):
    b1 = scipy.stats.binom(n, p1)
    b2 = scipy.stats.binom(n, p2)
    return math.fsum(b1.cdf(k-1)*b2.pmf(k) for k in range(1, n+1))

def find(p1, p2, conf, n=64, precision=0.01):
    n_lower = 0
    n_upper = None
    old = (None, None, None)
    while True:
        if n_lower > n or (n_upper and n_upper < n) or old == (n_lower, n, n_upper):
            raise ValueError("Failed.")
        old = (n_lower, n, n_upper)
        p_cur = prob(p1, p2, n)
        print(f"{n}: {p_cur}")
        if p_cur > conf:
            diff = max((int(precision*n), 1))
            n_test = max(0, n - diff)
            if prob(p1, p2, n_test) < conf:
                return n
            n_lower, n, n_upper = n_lower, n_lower + ((n - n_lower)//2), n
        else:
            if n_upper is None:
                n_lower, n = n, n*2
            else:
                n_lower, n, n_upper = n, n + ((n_upper - n)//2), n_upper
                
                
                
find(0.50, 0.51, 0.75)


import scipy.special as sp
p1 = 0.985
p2 = 0.995
conf = 0.95

2*((p1*(1-p1)+p2*(1-p2)) / (p2-p1) * sp.erfinv(2 * conf - 1))**2