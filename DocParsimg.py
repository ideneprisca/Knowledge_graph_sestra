import numpy as np
import xml.etree.cElementTree as Et
import re
from pathlib import Path
import string
import os


final = []
finalList=[]
character = "- "
characters_to_remove = "...."

data = []
finalList=[]
#characters_to_remove = {"punct":'- ' '\\' '•'}

#xml textual file loading, parsing and cleaning xml file

def listdirs(rootdir):
    global finalList
    fullname = []
    complete_text= []
    for directory in Path(rootdir).iterdir():
        if directory.is_dir():
            for filename in os.listdir(directory):
                if not filename.endswith('.xml'): continue
                file = os.path.join(directory, filename)
                tree = Et.parse(file)
                records = tree.getroot()
                for record in records.findall('record'):
                    record = record.find('page').text
                    data.append(record)
    return data


def data_cleaning(texte):
    for k in range(0,len(texte)):
        fin = re.sub('\s{2,}', ' ', str(texte[k]))
        cleanText1 = fin.replace(character, "").replace(characters_to_remove, "")
        cleanText2 = re.sub(r'(.|-|_|\|•)1+', '', cleanText1)
        finalList.append(cleanText2)
    return finalList


if __name__ == "__main__":
 rootdir = 'C:/Users/user/Desktop/thesis/sestra/bo0144'
 res = listdirs(rootdir)
 #print("final")
 print(data_cleaning(res))













