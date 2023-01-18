#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Created By  : Daniil Kazantsev <scientificsoftware@diamond.ac.uk>
# Created Date: 12/Janurary/2023
# version ='0.1'
# ---------------------------------------------------------------------------
"""Script that exposes httomolib functions from the list of given modules"""  

import re
import inspect
import yaml
import os
import importlib

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = os.path.dirname(os.path.abspath("httomolib_modules.yml"))
path_to_httomolib_modules = ROOT_DIR + '/httomolib_modules.yml'

discard_keys = ["data",
                "glob_stats",
                "comm_rank",
                "out_dir",
                "gpu_id",
                "angles",
                "flats",
                "darks"] # discard from parameters list
no_data_out_modules = ['save_to_images'] # discard data_out from certain modules

# open YAML file with httomolib modules exposed
with open(path_to_httomolib_modules, "r") as stream:
    try:
        httomolib_modules_list = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# a loop over httomolib modules
modules_no = len(httomolib_modules_list)
for i in range(modules_no):
   module_name = httomolib_modules_list[i]
   imported_module = importlib.import_module(str(module_name))
   methods_list = imported_module.__all__  # get all the methods in the module 
   methods_no = len(methods_list)

   # a loop over httomolib methods in the module
   for m in range(methods_no):
      method_name =  methods_list[m]
      get_method_params = inspect.signature(getattr(imported_module, methods_list[m]))
      # get method docstrings   
      get_method_docs = inspect.getdoc(getattr(imported_module, methods_list[m]))

      # put the parameters in the dictionary
      params_list = []
      params_dict = {}
      params_dict["data_in"] = 'tomo' # default dataset names
      if method_name not in no_data_out_modules:
         params_dict["data_out"] = 'tomo'   
      for k,v in get_method_params.parameters.items():
         if v is not None:
            append = True
            for x in discard_keys:
               if str(k) == x:
                  append = False
                  break
            if append:               
               if str(v).find("=") == -1 and  str(k) != "kwargs":
                  params_dict[str(k)] = 'REQUIRED'
               elif str(k) == "kwargs":
                  params_dict["#additional parameters"] = 'AVAILABLE'
               else:
                  params_dict[str(k)] = v.default
      
      params_list = [{module_name: {method_name: params_dict}}]

   
      # save the list as a YAML file
      path_dir = ROOT_DIR + '/' + module_name
      path_file = path_dir + '/' + str(method_name) + '.yaml'
      
      if not os.path.exists(path_dir):
         os.makedirs(path_dir)

      with open(path_file, 'w') as file:
         outputs = yaml.dump(params_list, file, sort_keys=False)
