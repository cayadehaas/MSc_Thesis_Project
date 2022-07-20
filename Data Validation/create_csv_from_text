import os
from tika import parser
import csv

with open('text_MBO_documents.csv', 'w', newline='', encoding='utf-8') as f:  # create csv file
    writer = csv.writer(f)
    writer.writerow(["filename", "description"])

    for root, dirs, files in os.walk('/MBO Raad/MBO Raad', topdown=False):
        for filename in files:
            if '.pdf' in filename:
                print(filename)
                rawText = parser.from_file(root + '/' + filename)
                clean_text = rawText['content'].replace('\n', '')
                writer.writerow([filename, clean_text])

