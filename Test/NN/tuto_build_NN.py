#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 17:48:50 2018

@author: fred
"""

import tensorflow as tf

# Create a variable with an initial value of 1
some_var = tf.Variable(1)

# Create op to run variable initializers
init_op = tf.global_variables_initializer()
 
# Create an op to replace the value held by some_var to 3
assign_op = some_var.assign(3)
 
# Set up two instances of a session
sess1 = tf.Session()
sess2 = tf.Session()

# Initialize variables in both sessions
sess1.run(init_op)
sess2.run(init_op)
print(sess1.run(some_var)) # Outputs 1

# Change some_var in session1
sess1.run(assign_op)
print(sess1.run(some_var)) # Outputs 3
print(sess2.run(some_var)) # Outputs 1

# Close sessions
sess1.close()
sess2.close()