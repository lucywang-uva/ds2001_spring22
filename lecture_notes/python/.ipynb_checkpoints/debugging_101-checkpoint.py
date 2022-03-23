#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 05:47:46 2022

@author: youmisuk
"""

# Debugging with print() and breakpoint()


# :::: Example: create a max function ::::
def max(lst):
    """Find the maximum integer value in a list"""
    max_num = 0 
    for num in lst:
        if num > max_num:
            max_num = num
    return max_num


print(max([-10,-5,-1])) # check the output.
    

# :::: Debugging 101: use the print statement ::::
def max(lst):
    """Find the maximum integer value in a list"""
    max_num = 0
    for num in lst:
        print('num:', num, 'maximum:', max_num)
        if num > max_num: 
            print('num larger; re-assigning value:', num)
            max_num = num
    return max_num

print(max([-10,-5,-1])) # check the output.


# quick fix
def max(lst):
    """Find the maximum integer value in a list"""
    max_num = -float('inf') # an infinitely small value
    for num in lst:
        print('num:', num, 'maximum:', max_num)
        if num > max_num: 
            print('num larger; re-assigning value:', num)
            max_num = num
    return max_num

print(max([-10,-5,-1])) # check the output.


# :::: Interactive debuggers: breakpoint() ::::
# Python’s breakpoint() built-in function is a new addition in version 3.7 
# Debugger commands (https://docs.python.org/3/library/pdb.html):
# h: help
# w: where
# n: next
# s: step (steps into function)
# c: continue
# l: list
# q: quit

def max(lst):
    """Find the maximum integer value in a list"""
    max_num = 0 
    for num in lst:
        breakpoint() # enter debug mode        
        if num > max_num:
            max_num = num
    return max_num

print(max([-10,-5,-1])) # check the output.

