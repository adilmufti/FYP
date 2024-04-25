'''
TO RUN THIS CODE YOU NEED TO OBTAIN YOUR OWN OPENAI API KEY AND ASSIGN THE 'API_KEY' VARIABLE TO IT
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

with open("formatted_fine_tuning_data.jsonl", "rb") as file:
    response = client.files.create(
        file=file,
        purpose="fine-tune"
    )

file_id = response.id
print(f"File Uploaded Successfully with ID: {file_id}")