import pandas as pd
import xml.etree.ElementTree as ET
from pdb import set_trace

tree = ET.parse('/mnt/c/Users/tnandi/Downloads/TCGA-LUAD_clinical.tar/00e01b05-d939-49e6-b808-e1bab0d6773b/nationwidechildrens.org_clinical.TCGA-J2-8192.xml')
root = tree.getroot()

data = []

# Iterate through XML tree and extract data
for child in root:
    row = {}
    for subchild in child:
        row[subchild.tag] = subchild.text
    data.append(row)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(data)

# Optionally, you can set the index if there's a unique identifier
# For example, if 'id' is a unique identifier for each record:
# df.set_index('id', inplace=True)

# Display the DataFrame
print(df)
set_trace()
