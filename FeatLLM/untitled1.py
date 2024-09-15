# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:39:41 2024

@author: jishu
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:34:18 2024

@author: jishu
"""

import os
import copy
import utils
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold

_NUM_QUERY = 5 # Number of ensembles
_SHOT = 4 # Number of training shots
_SEED = 0 # Seed for fixing randomness
_DATA = 'blood'
_API_KEY = 'sk-bmac9Ehl6LWaFJjtE06X7bw8oTM2S_AqibOA79k4X1T3BlbkFJuT_Q8tvaqbkY-fFNVqD1U4Lh1-wv5CmnaXH_9dKswA'


utils.set_seed(_SEED)
df, X_train, X_test, y_train, y_test, target_attr, label_list, is_cat = utils.get_dataset(_DATA, _SHOT, _SEED)
X_all = df.drop(target_attr, axis=1)

ask_file_name = './templates/ask_llm.txt'
meta_data_name = f"./data/{_DATA}-metadata.json"
templates, feature_desc = utils.get_prompt_for_asking(
    _DATA, X_all, X_train, y_train, label_list, target_attr, ask_file_name, 
    meta_data_name, is_cat, num_query=_NUM_QUERY
)
#print(templates[0])


_DIVIDER = "\n\n---DIVIDER---\n\n"
_VERSION = "\n\n---VERSION---\n\n"

rule_file_name = f'./rules/rule-{_DATA}-{_SHOT}-{_SEED}.out'
if os.path.isfile(rule_file_name) == False:
    print("yes") 
    results = utils.query_gpt(templates, _API_KEY, max_tokens=1500, temperature=0.5)
    print(results[0])
    with open(rule_file_name, 'w') as f:
        total_rules = _DIVIDER.join((results))
        f.write(total_rules)
else:
    with open(rule_file_name, 'r') as f:
        total_rules_str = f.read().strip()
        results = total_rules_str.split(_DIVIDER)

print(results[0]) 


parsed_rules = utils.parse_rules(results, label_list)

saved_file_name = f'./rules/function-{_DATA}-{_SHOT}-{_SEED}.out'    
if os.path.isfile(saved_file_name) == False:
    function_file_name = './templates/ask_for_function.txt'
    fct_strs_all = []
    for parsed_rule in tqdm(parsed_rules):
        fct_templates = utils.get_prompt_for_generating_function(
            parsed_rule, feature_desc, function_file_name
        )
        fct_results = utils.query_gpt(fct_templates, _API_KEY, max_tokens=1500, temperature=0)
        fct_strs = [fct_txt.split('<start>')[1].split('<end>')[0].strip() for fct_txt in fct_results]
        fct_strs_all.append(fct_strs)

    with open(saved_file_name, 'w') as f:
        total_str = _VERSION.join([_DIVIDER.join(x) for x in fct_strs_all])
        f.write(total_str)
else:
    with open(saved_file_name, 'r') as f:
        total_str = f.read().strip()
        fct_strs_all = [x.split(_DIVIDER) for x in total_str.split(_VERSION)]