from groq import Groq

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import json

# Initialize Groq client
client = Groq(api_key='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')  # Use your Groq API key



def map_attributes_to_columns(attributes, query , max_tokens = 1000):
    
    # Prepare the prompt
    prompt = f"""
    You are given a table with the following columns 
    {attributes}
    
    Here is a query about a person with their attribute values:
    {query}
    
    Por every column name please find it's corresponding number from the query.
    Return only a code which creates a dataframe using those values. The code should only contain lines which can be executed in a compiler.
    
    The code should start with import pandas as pd
   
    """
    
    # Send prompt to Groq API
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # Specify the model you want to use
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,         # Control randomness
        max_tokens=max_tokens,   # Control maximum tokens in the output
        top_p=0.9,               # Sampling control for nucleus sampling
        stream=False             # Change to True if you want streaming responses
    )

    # Access the content of the first choice in the response
    result = response.choices[0].message.content
    return result

def extract_code_from_string(code_str, keyword="import pandas as pd"):
    start_index = code_str.find(keyword)
    
    if start_index == -1:
        # Keyword not found
        return ""
    
    return code_str[start_index:-3]

def execute_code_string(code_str):
    # Define a local dictionary to capture the output
    local_vars = {}
    
    # Execute the code string
    exec(code_str, globals(), local_vars)
    
    # Retrieve and return the DataFrame if it exists
    return local_vars.get('df', None)



def GiveDF(attributes , query):
    # Map attributes to columns
    mapping = map_attributes_to_columns(attributes, query)
    mapping = extract_code_from_string(mapping)
    
    return execute_code_string(mapping)
    
def GiveGoodAnswer(classNames , answer , question , max_tokens = 1000):
    # Prepare the prompt
    prompt = f"""
    You are given with the target class name and the answer. 
    If the answer is 1 return a positive sentence.
    If the answer is 0 return a negative sentence. 
    
    
    classnames : {classNames}
    answer : {answer}
    question : {question}
    
   
    """
    
    # Send prompt to Groq API
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # Specify the model you want to use
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,         # Control randomness
        max_tokens=max_tokens,   # Control maximum tokens in the output
        top_p=0.9,               # Sampling control for nucleus sampling
        stream=False             # Change to True if you want streaming responses
    )
    
    # Access the content of the first choice in the response
    result = response.choices[0].message.content
    print(result)
    return 
