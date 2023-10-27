FILENAME = 'a_strict.txt'

import re
pattern = r'Accuracy/Eval: (\d+\.\d+)'
with open(FILENAME, 'r') as file:
    file_contents = file.read()
matches = re.findall(pattern, file_contents)
print(','.join(matches))
