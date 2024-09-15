# -*- coding: utf-8 -*-
_API_KEY = 'enter your api-key'

"""
Created on Sun Sep  8 17:34:18 2024

@author: jishu
"""

import os
import copy
import utils
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import ProperLLMResponse

from tqdm import tqdm
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold

_NUM_QUERY = 5 # Number of ensembles
_SHOT = 4 # Number of training shots
_SEED = 0 # Seed for fixing randomness
_DATA = 'blood'

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
    results = utils.query_gpt(templates, _API_KEY, max_tokens=1500, temperature=0.5)
    with open(rule_file_name, 'w') as f:
        total_rules = _DIVIDER.join(map(str,results))
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
        
# Get function names and strings
fct_names = []
fct_strs_final = []
for fct_str_pair in fct_strs_all:
    fct_pair_name = []
    if 'def' not in fct_str_pair[0]:
        continue

    for fct_str in fct_str_pair:
        fct_pair_name.append(fct_str.split('def')[1].split('(')[0].strip())
    fct_names.append(fct_pair_name)
    fct_strs_final.append(fct_str_pair)
#print(fct_strs_final[0][0])


executable_list, X_train_all_dict, X_test_all_dict = utils.convert_to_binary_vectors(fct_strs_final, fct_names, label_list, X_train, X_test)

class simple_model(nn.Module):
    def __init__(self, X):
        super(simple_model, self).__init__()
        # Initialize weights based on X
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(x_each.shape[1], 1) / x_each.shape[1])
            for x_each in X
        ])
        
    def forward(self, x):
        x_total_score = []
        for idx, x_each in enumerate(x):
            # Ensure idx is within range of self.weights
            if idx >= len(self.weights):
                raise IndexError(f'Index {idx} is out of range for self.weights with length {len(self.weights)}')
            
            x_score = x_each @ torch.clamp(self.weights[idx], min=0)
            x_total_score.append(x_score)
        
        x_total_score = torch.cat(x_total_score, dim=-1)
        return x_total_score
    
test_outputs_all = []
multiclass = True if len(label_list) > 2 else False
y_train_num = np.array([label_list.index(k) for k in y_train])
y_test_num = np.array([label_list.index(k) for k in y_test])

def train(X_train_now, label_list, shot):
    criterion = nn.CrossEntropyLoss()                
    if shot // len(label_list) == 1:
        model = simple_model(X_train_now)
        opt = Adam(model.parameters(), lr=1e-2)
        for _ in range(200):                    
            opt.zero_grad()
            outputs = model(X_train_now)
            preds = outputs.argmax(dim=1).numpy()
            acc = (np.array(y_train_num) == preds).sum() / len(preds)
            if acc == 1:
                break
            loss = criterion(outputs, torch.tensor(y_train_num))
            loss.backward()
            opt.step()
    else:
        if shot // len(label_list) <= 2:
            n_splits = 2
        else:
            n_splits = 4

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
        model_list = []
        for fold, (train_ids, valid_ids) in enumerate(kfold.split(X_train_now[0], y_train_num)):
            model = simple_model(X_train_now)
            opt = Adam(model.parameters(), lr=1e-2)
            X_train_now_fold = [x_train_now[train_ids] for x_train_now in X_train_now]
            X_valid_now_fold = [x_train_now[valid_ids] for x_train_now in X_train_now]
            y_train_fold = y_train_num[train_ids]
            y_valid_fold = y_train_num[valid_ids]

            max_acc = -1
            for _ in range(200):                    
                opt.zero_grad()
                outputs = model(X_train_now_fold)
                loss = criterion(outputs, torch.tensor(y_train_fold , dtype=torch.long))
                loss.backward()
                opt.step()

                valid_outputs = model(X_valid_now_fold)
                preds = valid_outputs.argmax(dim=1).numpy()
                acc = (np.array(y_valid_fold) == preds).sum() / len(preds)
                if max_acc < acc:
                    max_acc = acc 
                    final_model = copy.deepcopy(model)
                    if max_acc >= 1:
                        break
            model_list.append(final_model)

        sdict = model_list[0].state_dict()
        for key in sdict:
            sdict[key] = torch.stack([model.state_dict()[key] for model in model_list], dim=0).mean(dim=0)

        model = simple_model(X_train_now)
        model.load_state_dict(sdict)
    return model

test_outputs_all = []
multiclass = True if len(label_list) > 2 else False
y_train_num = np.array([label_list.index(k) for k in y_train])
y_test_num = np.array([label_list.index(k) for k in y_test])


for i in executable_list:
    X_train_now = list(X_train_all_dict[i].values())
    X_test_now = list(X_test_all_dict[i].values())
    
    # Train
    trained_model = train(X_train_now, label_list, _SHOT)

    # Evaluate
    test_outputs = trained_model(X_test_now).detach().cpu()
    test_outputs = F.softmax(test_outputs, dim=1).detach()
    result_auc = utils.evaluate(test_outputs.numpy(), y_test_num, multiclass=multiclass)
    #print("AUC:", result_auc)
    test_outputs_all.append(test_outputs)
    
test_outputs_all = np.stack(test_outputs_all, axis=0)
ensembled_probs = test_outputs_all.mean(0)
result_auc = utils.evaluate(ensembled_probs, y_test_num, multiclass=multiclass)
print("Ensembled AUC:", result_auc)

def infer_binary_vectors(fct_strs_all, fct_names, label_list, X_train, X_test, X_new):
    # Ensure X_new is a DataFrame
    if isinstance(X_new, np.ndarray):
        X_new = pd.DataFrame(X_new, columns=X_train.columns)
    
    X_new_dict = {}
    for label in label_list:
        X_new_dict[label] = {}

    for i in range(len(fct_strs_all)): # len(fct_strs_all) == # of trials for ensemble
        try:
            # Match function names with each answer class
            fct_idx_dict = {}
            for idx, name in enumerate(fct_names[i]):
                for label in label_list:
                    label_name = '_'.join(label.split(' '))
                    if label_name.lower() in name.lower():
                        fct_idx_dict[label] = idx

            # If the number of inferred rules are not the same as the number of answer classes, skip the trial
            if len(fct_idx_dict) != len(label_list):
                continue

            # Apply the functions to X_new
            for label in label_list:
                fct_idx = fct_idx_dict[label]
                exec(fct_strs_all[i][fct_idx].strip('` "'))
                func = locals()[fct_names[i][fct_idx]]
                
                # Convert X_new to binary vectors
                X_new_each = func(X_new).astype('int').to_numpy()
                
                # Store results in a dictionary
                X_new_dict[label] = torch.tensor(X_new_each).float()

        except Exception as e: # If an error occurred during function application, skip the trial
            print(f"Error processing trial {i}: {e}")
            continue

    return X_new_dict

def infer_from_model(model, X_new_dict):
    model.eval()  # Ensure the model is in evaluation mode
    
    with torch.no_grad():
        # Process each entry in X_new_dict
        outputs = [model(X_new_dict[label]) for label in X_new_dict]
        
        # Convert outputs to probabilities
        probs = [F.softmax(output, dim=1).numpy() for output in outputs]
        
        # Aggregate predictions
        final_predictions = np.mean(probs, axis=0)
    
    return final_predictions

def get_model_predictions(model, X_new_dict):
    
        # Ensure the model is in evaluation mode
    model.eval()
    X_test_now = list(X_new_dict.values())
        
        # Get model predictions for X_new_dict
    with torch.no_grad():
            # Collect predictions from all labels
        # Evaluate
        test_outputs = trained_model(X_test_now).detach().cpu()
        test_outputs = F.softmax(test_outputs, dim=1).detach()
    
        # Aggregate across all models
   
    return test_outputs

# Example usage with `executable_list`, `X_train_all_dict`, `X_test_all_dict`:
def process_inference(models, fct_strs_all, fct_names, label_list, X_train, X_test, X_new, executable_list):
    X_new_dict = infer_binary_vectors(fct_strs_all, fct_names, label_list, X_train, X_test, X_new)

    # Iterate over the executable list and process new data
    for i in range(1):
        X_new_now = X_new_dict  # Use X_new_dict as the input for inference
        
        # Get predictions for the new data
        predictions = get_model_predictions(models, X_new_now)
        
        # Example of converting predictions to class labels
        predicted_classes = np.argmax(predictions, axis=1)  # Get the class with the highest probability
        print(f"Predicted classes for trial {i}: {predicted_classes}")
        return predicted_classes

# Example attributes and query
attributes = """
Recency
Frequency
Monetary
Time
"""

class_values = {"yes" : 1 , "no" : 0}

query = "What will be the class of a person having frequency 2, recency 5 ,monetary 89, and time 1."

print(query)

df = ProperLLMResponse.GiveDF(attributes, query)
predicted_classes = process_inference(trained_model, fct_strs_all, fct_names, label_list, X_train, X_test, df, executable_list)

ProperLLMResponse.GiveGoodAnswer(class_values, predicted_classes , query)


