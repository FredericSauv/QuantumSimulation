#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 21:02:09 2018

@author: frederic
"""
import os
 
from multiprocessing import Process
 
def doubler(number):
    """ A doubling function that can be used by a process
    """
    result = number * 2
    proc = os.getpid()
    print('{0} doubled to {1} by process id: {2}'.format(
        number, result, proc))
 
if __name__ == '__main__':
    numbers = [5, 10, 15, 20, 25]
    procs = []
    proc = Process(target=doubler, args=(5,))
 
    for index, number in enumerate(numbers):
        proc = Process(target=doubler, args=(number,))
        procs.append(proc)
        proc.start()
 
    proc = Process(target=doubler, name='Test', args=(2,))
    proc.start()
    procs.append(proc)
 
    for proc in procs:
        proc.join()