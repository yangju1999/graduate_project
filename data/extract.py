import yaml 
import json 
import random 

data = {}
data['events'] = []

with open('k8saudit.yaml') as f:
    k8s_rules = yaml.load(f, Loader=yaml.FullLoader)
    for i in range(len(k8s_rules)):
        if k8s_rules[i] != None and 'output' in list(k8s_rules[i].keys()):
            end_index = k8s_rules[i]['output'].find('(')
            temp = {'output': k8s_rules[i]['output'][:end_index-1]+'.'}
        
            data['events'].append(temp)

            
            
with open('default.yaml') as f:
    default_rules = yaml.load(f, Loader=yaml.FullLoader)
    for i in range(len(default_rules)):
        if default_rules[i] != None and 'output' in list(default_rules[i].keys()):
            end_index = default_rules[i]['output'].find('(')
            temp = {'output': default_rules[i]['output'][:end_index-1] + '.'}
        
            data['events'].append(temp)

random.shuffle(data['events'])

new_data = {}
new_data['event_sequence'] = []


for i in range(len(data["events"]) - 2):
    temp = {}
    for count in range(3):
        temp[str(count+1)] = data["events"][i+count]["output"]
    new_data['event_sequence'].append(temp)


file_path = "./data.json"

with open(file_path, 'w') as outfile:
    json.dump(new_data, outfile, indent = 4)
