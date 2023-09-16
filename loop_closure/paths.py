#!/usr/bin/env python3

import os
import sys

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def get_root_path():
    return os.path.join(get_script_path(),'..')

def get_new_college_dataset_path():
    return os.path.join(get_root_path(),'data','new_college')

def get_confusion_matrix_path():
    return os.path.join(get_root_path(),'build','loop_closure','confusion_matrix.txt')
