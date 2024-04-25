'''
TO RUN THIS CODE YOU NEED TO OBTAIN YOUR OWN OPENAI API KEY AND ASSIGN THE 'API_KEY' VARIABLE TO IT.

THEN FIRSTLY, YOU NEED TO RUN THE UPLOADING_FINET.PY FILE TO OBTAIN A FILE ID. THEN TAKE THIS FILE ID
AND ASSIGN THE 'FILE_ID' VARIABLE TO THE FILE ID THAT GETS PRINTED OUT FROM RUNNING THAT FILE

THEN THIS FILE IS RUNNABLE

NOTE: INSTRUCTIONS ON HOW TO CHECK THE FINE-TUNING STATUS ARE LISTED AT THE BOTTOM OF THE FILE
'''

import openai
from openai import OpenAI

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)

api_key = 'INSERT API KEY'

client = openai.OpenAI(api_key=api_key)


file_id = 'INSERT FILE ID'
model_name = 'gpt-3.5-turbo-1106'

response = client.fine_tuning.jobs.create(
    training_file=file_id,
    model=model_name
)

job_id = response.id
print(f"Fine-Tuning job created successfully with ID: {job_id}")



'''
# CHECK JOB STATUS
# print(client.fine_tuning.jobs.retrieve("FILE ID HERE"))

# ALTERNATIVELY, VISIT: https://platform.openai.com/finetune
'''