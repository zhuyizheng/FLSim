import argparse
import os

parser = argparse.ArgumentParser(description="extract accuracy")

parser.add_argument("input", type=str, help="input file path")
args = parser.parse_args()

path = args.input
file_list = os.listdir(path)

def acc_name_converter(name):
    return name[:-4] + '_acc.txt'

for name in file_list:
    if name[-4:] != '.txt':
        continue
    if name[-8:] == '_acc.txt':
        continue

    FILENAME = os.path.join(path, name)
    # assert FILENAME.endswith('.txt'), "wrong input file format!"
    OUTPUTFILENAME = os.path.join(path, acc_name_converter(name))

    import re
    pattern = r'Accuracy/Eval: (\d+\.\d+)'
    with open(FILENAME, 'r') as file:
        file_contents = file.read()
    matches = re.findall(pattern, file_contents)
    print(f'reading {name[:-4]}:')
    print(','.join(matches))

    # Open the output file for writing
    with open(OUTPUTFILENAME, 'w') as output_file:
        # Write each match to the output file on a separate line
        for match in matches:
            output_file.write(match + '\n')