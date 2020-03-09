import csv
import os
from os import path

csv.register_dialect("hashes", delimiter="#")

class Dict(dict):
    def __missing__(self, key):
        return 0

text_dict = Dict()
complexity_dict = Dict()
text_with_dup_dict = Dict()

def write_to_csv(dict1, dict2, file_name):
    with open(file_name, 'w', newline='', encoding="utf8") as csvfile:
        writer = csv.writer(csvfile, dialect="hashes")
        keys = dict1.keys()
        for k in keys:
            writer.writerow((dict2[k].replace('"', '').replace('[', '').replace(']', '').replace('-', ' '), dict1[k]))

def read_from_csv(file_name):
    with open(file_name, mode='r', encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="#")
        line_count = 0
        for row in csv_reader:
            line_count += 1

            complexity_dict[line_count] = row[1]
            text_dict[line_count] = row[0]
            # print(row[0])
        print(line_count)

if __name__ == "__main__":
    d = os.getcwd()
    print(d)
    textPath = path.join(d, 'Train_all_types_hashes.csv')
    print(textPath)
    read_from_csv(textPath)

    print('---------------------Train_all_types.csv------------------------')
    print(f'Processed {len(text_dict)} lines.')
    # write_to_csv( complexity_dict, text_dict, 'Train_all_types_hashes.csv')