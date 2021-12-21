#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:36:34 2021

@author: orynaivashtenko
"""
import numpy as np
import os


def is_int(string):
    assert type(string)==str, 'Input must be string!'
    if string.isdecimal():
        return True
    elif string[0]=='-' and (string[1:]).isdecimal():
        return True
    else:
        return False
    
    
def is_simple_num(string):
    '''
    Checks whether the function is a simple number, that is float
    not in scientific"e+..." format
    '''
    
    if is_int(string):
        return True
    elif string.count('.')==1 and is_int(string.replace('.', '')):
        return True
    else:
        return False
    
    
def is_float_number(string):
    string = string.lower()
    if is_simple_num(string):
        return True
    elif ((string.count('e')==1 and is_simple_num(string.replace('e', '')))
          or (string.count('e-')==1 and is_simple_num(string.replace('e-', '')))
          or (string.count('e+')==1 and is_simple_num(string.replace('e+', '')))):
        return True
    else:
        return False

    
def is_float(string):
    string = string.lower()
    if is_float_number(string):
        return True
    elif (string=='nan' or string=='inf' or string=='-inf'):
        return True
    else:
        return False
    

def is_bool(string):
    if (string=='True' or string=='False'):
        return True
    else:
        return False
    
    
def is_boolable(string):
    if (string=='True' or string=='False' 
        or string=='0' or string=='1'
        or string=='0.' or string=='1.'):
        return True
    else:
        return False
    
        

def variables_from_file(file):
    """
    Opens the file with a list of variable names and values. Recognizes the data
    types of the values. Creates variables with corresponding names and assigns 
    them the values. Format of the file (example):
        dummy_coefficient = 1.
        curse='hann'
        
        best_number=13
        is_life_great = True
        
    Parameters
    ----------
    file : str
        Name (full path) of the file from which the variables should be read.
    Returns
    -------
    None.
    
    Creates 
    -------
    The variables and assigns values
    """
    f = open(file, 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        if '=' in line:
            text = line.split('#')[0] #remove comments
            text = text.split('\\')[0] #remove endlines
            name, val = text.split('=')[0:2]
            name = name.strip()
            val = val.strip()
            if ((name!='') and (val!='')):
                if val=='None':
                    val=None
                elif val=='True':
                    val=True
                elif val=='False':
                    val=False
                elif is_int(val):
                    val = int(val)
                elif is_float(val):
                    val = float(val)
                else:
                    assert (((val[0]=='"') and (val[-1]=='"')) 
                            or ((val[0]=="'") and (val[-1]=="'"))), 'Undefined type of value: not int, bool, float, str or None! ( line "'+str(line)+'" )'
                    val = val[1:-1]
                    
                assert ((65<=ord(name[0])<=90) or (97<=ord(name[0])<=122) or name[0]=='_') , 'Incorrect variable name specified!'
                if len(name)>1:
                    for e in name[1:]:
                        assert ((48<=ord(e)<=57) or (65<=ord(e)<=90) or (97<=ord(e)<=122) or e=='_') , 'Incorrect variable name specified!'
                globals()[name] = val
                

def read_lists_from_file(file, location=None, dimension=2, to_array=True):
    """
    Reads an array (numpy.array or list of float, maximum 2d) printed to the txt file 
    named name and returns the output in form of list.

    Parameters
    ----------
     name : str
        Name of the file ('.txt' is not needed). 
    location : str
        Location where to create the file. If None, the current local directory
        will be used

    Returns
    -------
    res : list
        List (possibly with nested lists) of data read from the file.

    """
    if location is not None:
        file = os.path.join(location, file)
    else:
        if file[-4:]!='.txt':
            file = file +'.txt'
    file = open(file, 'r')
    res = file.read()
    if dimension<3:
        res = res.replace('array', '')
        res = res.replace('(', '')
        res = res.replace(')', '')
        res = res.replace(',', ' ')
        res = (res.split('] ['))
        for i in range(len(res)):
            res[i] = res[i].replace(']','')
            res[i] = res[i].replace('[','')
            res[i] = res[i].replace('\n','')
            res[i] = list(map(float,res[i].split()))
        if len(res)==1:
            res = res[0]
    elif dimension==3:
        res = res.replace('array', '')
        res = res.replace('(', '')
        res = res.replace(')', '')
        res = res.replace(',', ' ')
        res = res.replace('  ', ' ') #important to do in because 
        #otherwise ']  [' with two spaces can appear (which we can't see)
        res = (res.split(']] [['))
        for j in range(len(res)):
            res[j] = (res[j].split('] ['))
            for i in range(len(res[j])):
                res[j][i] = res[j][i].replace(']','')
                res[j][i] = res[j][i].replace('[','')
                res[j][i] = res[j][i].replace('\n','')
                if to_array:
                    res[j][i] = np.array(list(map(float,res[j][i].split())))
                else:
                    res[j][i] = list(map(float,res[j][i].split()))
        file.close()
    return res

                
def read_csv_file(file, location=None, to_type=None):
    if location is not None:
        file = os.path.join(location, file)
    file = open(file, 'r')
    res = file.read()
    file.close()
    res = res.split(',')
    res = list(map(lambda x: x.strip(), res))
    if to_type is None:
        isfloat = True
        isint = True
        isbool = True
        for entry in res:
            if not is_int(entry):
                isint = False
            if not is_float(entry):
                isfloat = False
            if not is_bool(entry):
                isbool = False
            if (not isint) and (not isfloat) and (not isbool):
                break
        if isbool:
            to_type=bool
        if isint:
            to_type=float
        if isfloat:
            to_type=float
            
    if to_type==int or to_type==float or to_type==bool:    
        res = list(map(to_type, res))
    elif to_type==str:
        pass
    else:
        raise AssertionError('Incorrect `to_type` specified!')
    return res
    