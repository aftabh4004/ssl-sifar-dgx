import os
import glob

directory_path = './output/'

def process(file_path):
    import json
    json_objects = []

    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                json_objects.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")

    file.close()
    sup_acc = 0
    unsup_start = 0
    max_acc = 0
    max_acc_epoch = 0
    for obj in json_objects:
        if max_acc < obj['test_acc1']:
            max_acc = obj['test_acc1']
            max_acc_epoch = obj['epoch']

        if obj['epoch'] == 24: sup_acc = obj['test_acc1']
        if obj['epoch'] == 25: unsup_start = obj['test_acc1']
    
    with open('results.csv', 'a') as out:
        print(f"{file_path},{sup_acc},{unsup_start - sup_acc},{max_acc},{max_acc_epoch}", file=out)


with open('results.csv', 'w') as out:
    print(f"file_path,sup_acc,drop,max_acc,max_acc_epoch", file=out)
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            process(file_path)
