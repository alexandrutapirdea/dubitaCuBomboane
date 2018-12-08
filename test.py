# import csv
# spamReader = csv.reader(open('good.csv', newline=''), delimiter=' ', quotechar='|')
# for row in spamReader: print(', '.join(row[0]))

import csv
import string
outputfile = open('placelist.txt','w')
with open("good.csv") as csvFile:
    reader = csv.reader(csvFile, delimiter='\t', quoting=csv.QUOTE_NONE)
    reader_data = []
    for row in reader:
        reader_data.append(row)

    def remove_punctuation(from_text):
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in from_text]
        return stripped

    myList = []

    for element in reader_data:
        myList.append(remove_punctuation(element))

    # for element in myList:
    #     str1 = ''.join(element)
    #     outputfile.write(str1 + '\t');


myList = [[k.lower()] for l in myList for k in l]

for element in myList:
    print(element)