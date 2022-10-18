import tomopy.prep.stripe
import re
import inspect
import yaml
import os 

ROOT_DIR = os.path.dirname(os.path.abspath(".root_httomo"))

module_name = tomopy.prep.stripe
method_substring = 'stripe'

# extracting all methods from the given module 
methods_list = []
# attribute is a string representing the attribute name
for attribute in dir(module_name):
   # Get the attribute value
   attribute_value = getattr(module_name, attribute)
   # Check that it is callable
   if callable(attribute_value):
      # Filter 
      if attribute.startswith('__') == False and attribute.startswith('_') == False:
         for match in re.finditer(method_substring, attribute):
            methods_list.append(attribute)

total_methods = len(methods_list)

for m in range(total_methods):
   method_name = methods_list[m]
   # get function parameters with default values
   get_method_params = inspect.signature(locals()[method_name])
   # get function docstrings
   get_method_docs = locals()[method_name].__doc__

   # put the parameters in the dictionary
   params_dict = {str(method_name) : []}
   grow_list = []
   for k,v in get_method_params.parameters.items():
      if v is not None and str(k) != 'tomo':
         grow_list.append({str(k): str(v).split('=')[1::2][0]})
   params_dict[str(method_name)] = grow_list


   path_dir = ROOT_DIR + '/templates/tasks/stripe_removal/'
   path_file = path_dir + str(method_name) + '.yaml'

   # save the dictionary as a YAML file
   with open(path_file, 'w') as file:
      outputs = yaml.dump(params_dict, file)