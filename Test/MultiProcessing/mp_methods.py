#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 11:55:16 2018

@author: fred
"""

import os
import multiprocessing as mp

def func_simple(x):
    print("simple func used with %s" % x)
    return x**2



class A:
    def __init__(self):
        self.a = 2
    
    @staticmethod
    def func_staticmeth(x):
        return x**2
    
    @classmethod
    def func_classmeth(cls, x):
        return x**2
    
    def func_alone(self, x):
        return x**2
    
    def func_attribute(self, x):
        print("for pid = %s object %s has attribute a is %s" % (os.getpid(), str(hex(id(self))), self.a))
        return x**self.a
    
    def func_tricky(self, x):
        res = x ** self.a
        print("for pid = %s object %s has attribute a is %s" % (os.getpid(), str(hex(id(self))), self.a))
        self.a *= 2 
        #print(os.getpid())
        print("for pid = %s object %s has attribute a is %s" % (os.getpid(), str(hex(id(self))), self.a))
        return res

print("There are %d CPUs on this machine" % mp.cpu_count())

number_processes = 4
pool = mp.Pool(number_processes)
tasks = [1,10,100,1000, 2, 20, 200, 2000, 4,40,400,4000]
#total_tasks = len(tasks)
#res = pool.map(func_simple, tasks)

#func_test = A.func_staticmeth
#print(pool.map(func_test, tasks))
#
#Ainstance = A()
#func_test = Ainstance.func_classmeth
#print(pool.map(func_test, tasks))
#
#func_test = Ainstance.func_alone
#print(pool.map(func_test, tasks))

#print('########### FUNC ATTRIBUTE #########')
#Ainstance = A()
#print(str(hex(id(Ainstance))))
#func_test = Ainstance.func_attribute
#print(pool.map(func_test, tasks))


#
#print('########### FUNC TRICKY #########')
Ainstance = A()
print(str(hex(id(Ainstance))))
func_test = Ainstance.func_tricky
print(pool.map(func_test, tasks))



#pool.close()
#pool.join()