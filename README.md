# Knowledge_graph_sestra
Master thesis project
**From unstructured text to knowledge graph**

The project is a complete end-to-end solution for generating knowledge graphs from unstructured data. NER can be run on input by either NLTK, Spacy or Stanford APIs. Optionally, coreference resolution can be performed which is done by python wrapper to stanford's core NLP API. Relation extraction is then done using stanford's open ie. Lastly, post-processing is done to get csv file which can be uploaded to graph commons to visualize the knowledge graph.

More details can be found in the Approach folder.



**Running the code**

Clone Repository
Ensure your system is setup properly (Refer Setup instructions below)
Put your input data files (.txt) in data/input
Run knowledge_graph.py
python3 knowledge_graph.py spacy You can provide several arguments to knowledge_graph.py. For a more detailed list, refer the running knowledge_graph.py section below
Run relation_extractor.py python3 relation_extractor.py
Run create_structured_csv python3 create_structured_csv.py
The resultant csv is available in data/results folder



**Setup**

The following installation steps are written w.r.t. linux operating system and python3 language.

Create a new python3 virtual environment:
python3 -m venv <path_to_env/env_name>
Switch to the environment:
source path_to_env/env_name/bin/activate
Install Spacy:
pip3 install spacy
Install en_core_web_sm model for spacy:
python3 -m spacy download en_core_web_sm
Install nltk:
pip3 install nltk
Install required nltk data. Either install required packages individually or install all packages by using
python -m nltk.downloader all
Refer: https://www.nltk.org/data.html
Install stanfordcorenlp python package:
pip3 install stanfordcorenlp
Download and unzip stanford-corenlp-full:
https://stanfordnlp.github.io/CoreNLP/download.html
Download and setup stanford ner: https://nlp.stanford.edu/software/CRF-NER.shtml#Download as described in NLTK documentation: http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford (Not required if already present due to git clone)
Download and unzip stanford open-ie (Not required if already present due to git clone)
Install python-tk:
sudo apt-get install python3-tk
Install pandas:
pip3 install pandas
