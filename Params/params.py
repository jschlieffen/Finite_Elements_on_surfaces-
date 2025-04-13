#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:48:03 2025

@author: jschlieffen
"""

import configparser as cfg
import os
import sys
#sys.path.append(os.path.abspath('../logscripts/'))
from log_msg import *

# =============================================================================
# This file reads the params from the config file and checks the validation
# =============================================================================

class Params:
    
    # =============================================================================
    # Here are the default values of the params that are not necessarily required   
    # =============================================================================
    def __init__(self,config_file):
        self.show_progress_bar = 0
        self.config = cfg.ConfigParser()
        self.config.read(config_file)
        self.get_params(config_file)
        
    # =============================================================================
    # only for debug purposes    
    # =============================================================================
    def __str__(self):
        attributes = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attributes})"
    
    
    # =============================================================================
    # gets the params for the different params and defines if they are required 
    # or not 
    # =============================================================================
    def get_params(self, config_file):        
        self.show_progress_bar = self.get_value('monotoring', 'show_progress_bar', 'bool', required=True)
        self.show_ressource_usage = self.get_value('monotoring', 'show_ressource_usage', 'bool', required=True)
        self.calculation_of_error_estimates = self.get_value('error_estimates', 'calculation_of_error_estimates', 'bool',required=True)
        
    
    def get_value(self, section, param, par_type, var=None, required=False):
        val = os.getenv(param)
        
        if val is not None:
            if par_type=='int':
                return int(val)
            return val
        elif self.config.has_option(section, param):
            if par_type == 'int':
                return self.config.getint(section, param)
            elif par_type == 'bool':
                return self.config.getboolean(section, param)
            else:
                return self.config.get(section, param)
        elif required:
            logger.critical(f'required Param not set: {param}')
            sys.exit(1)
        else:
            logger.warning(f'not required Param not set: {param}')
            return var
        
    # =============================================================================
    # Checks the validation of certain parameters. Currently every validation has 
    # to be added by hand    
    # =============================================================================
    def validation_params(self):
        logger.warning('Params are currently not checked for validation')
# =============================================================================
# only for dev purpose
# =============================================================================
def main():
    Par_ = Params('config.cfg')
    Par_.validation_params()
    print('\n')
    print(Par_)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

if __name__ == '__main__':
    main()