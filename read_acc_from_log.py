FILENAME = 'a_strict.txt'
OUTPUTFILENAME = 'a_strict_acc.txt'
import re
pattern = r'Accuracy/Eval: (\d+\.\d+)'
with open(FILENAME, 'r') as file:
    file_contents = file.read()
matches = re.findall(pattern, file_contents)
print(','.join(matches))

# Open the output file for writing
with open(OUTPUTFILENAME, 'w') as output_file:
    # Write each match to the output file on a separate line
    for match in matches:
        output_file.write(match + '\n')