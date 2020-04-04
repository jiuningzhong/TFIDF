import csv
import os
import re

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
            writer.writerow((dict2[k], dict1[k]))

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

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'!"#%&(),:;<=>@]_`}~', '', text)  # *+/

    # text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    # text = re.sub(r'\w*\d\w*', '', text)
    # text = re.sub(r'http\S+', 'urllink', text, flags=re.S)

    # remove http/https, punctuation, and remove space at the beginning and the end
    text = text.replace('/', ' ').replace('<', ' ').replace('>', ' ').replace(';', ' ').replace(':', ' ') \
        .replace(')', ' ').replace('!', ' ').replace('?', ' ').replace('.', ' ').replace(',', ' ') \
        .replace('[', '').replace(']', '').replace('-', '').replace('(', ' ') \
        .replace('$', ' ').replace('\n', '').replace('\t', '').replace('\r', '')

    # print('before strip():' + text)

    text = (" ".join(text.split())).strip()
    # print('after strip():' + text)
    # split text without spaces into list of words and concatenate string in a list to a single string
    # text = ' '.join(wn.split(text))

    # spell check

    return text

if __name__ == "__main__":
    d = os.getcwd()
    print(d)
    textPath = path.join(d, 'Train_all_types_hashes.csv')
    print(textPath)
    read_from_csv(textPath)

    print('---------------------Train_all_types.csv------------------------')
    print(f'Processed {len(text_dict)} lines.')
    # write_to_csv( complexity_dict, text_dict, 'Train_all_types_hashes.csv')