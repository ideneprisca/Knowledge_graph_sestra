import warnings
import pandas as pd
import spacy
from DocParsimg import listdirs, data_cleaning

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)




rootdir = 'C:/Users/user/Desktop/thesis/sestra/bo0144'

nlp = spacy.load("it_nerIta_trf")

nlp.add_pipe('opentapioca', last=True) #debole
#nlp.add_pipe("entityfishing", last=True)#troppo ambiguità nel riconoscere le entità

entities_columns = pd.DataFrame(columns=['Text', 'entity_Label', 'entity_kb_id'])
dictionary = {"labels": "CARDINAL" "PERCENT" "QUANTITY" "ORDINAL" "LANGUAGE"}
pageContent = listdirs(rootdir)
allContent = data_cleaning(pageContent)
for k in range(len(allContent)):
    doc = nlp(allContent[k])
    for ent in doc.ents:
        if ent.label_ not in dictionary["labels"]:

           #write in entities and link in csv file
            '''
              parameters
            Text': ent.text
            'entity_Label': ent.label_, 'entity_kb_id': "https://www.wikidata.org/entity/" + ent.kb_id_ '''

            entities_columns = pd.concat([entities_columns, pd.DataFrame.from_records([{'Text': ent.text, 'entity_Label': ent.label_, 'entity_kb_id': "https://www.wikidata.org/entity/" + ent.kb_id_}])]).drop_duplicates()
    entities_columns.to_csv('./data/Output/Ner/entities_linked.csv')






