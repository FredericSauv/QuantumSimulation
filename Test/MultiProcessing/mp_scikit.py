
from sklearn.externals.joblib import Parallel, delayed
import itertools
import time
from multiprocessing import Process, Pipe


def factorial(num, the_dict):
    """ just return factorial(num)
    """
    print('[%s] Processing num: %d' % (time.strftime('%Y-%m-%d %T'), num))
    result = 1
    for i in range(1, num):
        result *= i
    return result

def multi_factorial(num_range, the_dict, pipe=None):
    """
    """
    result = []
    for num in num_range:
        result.append(factorial(num, the_dict))
    if pipe:
        pipe.send(result)
    else:
        return result

def _batch_iter(iterable, size):
    """ Slice the iterable into a genrator of chunks with size size
    """
    iterable = iter(iterable)
    batch = list(itertools.islice(iterable, size))
    while batch:
        yield batch
        batch = list(itertools.islice(iterable, size))

def run_scikit_parallel(huge_dict):
    result = Parallel(n_jobs=4)(delayed(multi_factorial)(nums, huge_dict)
                                        for nums in _batch_iter(range(100), 25))
    result = list(itertools.chain(*result))
    print(result)

def run_normal_parallel(huge_dict):
    workers = []
    pipes = []
    for i in range(4):
        parent, child = Pipe()
        worker = Process(target=multi_factorial, args=(range(i*25, (i+1)*25), huge_dict, child))
        pipes.append(parent)
        worker.start()

    result = []
    for pipe in pipes:
        result.extend(pipe.recv())
    for worker in workers:
        worker.terminate()
    print result

def main(idx):
    huge_dict = {}
    for string in itertools.combinations('abcdefghijklmnopqrstuvwxyz', 6):
        if 'a' in string:
            continue
        huge_dict[string] = 1
    print '[%s] Finish creating dictionary!' % time.strftime('%Y-%m-%d %T')
    if idx == 0:
        print 'Running normal parallel'
        run_normal_parallel(huge_dict)
    else:
        print 'Running scikit parallel'
        run_scikit_parallel(huge_dict)

if __name__ == '__main__':
    import sys
    idx = 0
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    main(idx)
    
    
    
    
    
    
    
    
    
    
    
from multiprocessing import Pool, TimeoutError
import time
import os

def f(x):
    return x*x

if __name__ == '__main__':
    # start 4 worker processes
    with Pool(processes=4) as pool:

        # print "[0, 1, 4,..., 81]"
        print(pool.map(f, range(10)))

        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f, range(10)):
            print(i)

        # evaluate "f(20)" asynchronously
        res = pool.apply_async(f, (20,))      # runs in *only* one process
        print(res.get(timeout=1))             # prints "400"

        # evaluate "os.getpid()" asynchronously
        res = pool.apply_async(os.getpid, ()) # runs in *only* one process
        print(res.get(timeout=1))             # prints the PID of that process

        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        print([res.get(timeout=1) for res in multiple_results])

        # make a single worker sleep for 10 secs
        res = pool.apply_async(time.sleep, (10,))
        try:
            print(res.get(timeout=1))
        except TimeoutError:
            print("We lacked patience and got a multiprocessing.TimeoutError")

        print("For the moment, the pool remains available for more work")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")