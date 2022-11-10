import pandas as pd
import spacy


def ner(doc):

    #nlp = spacy.load("it_nerIta_trf")

    nlp = spacy.load("it_nerIta_trf")
    nlp.add_pipe('opentapioca', last=True)
    doc = nlp(doc)
    return [(X.text,X.label_, X.kb_id_) for X in doc.ents]


def ner_to_dict(ner):
    """
    Expects ner of the form list of tuples
    """
    ner_dict = {}
    for tup in ner:
        ner_dict[tup[0]] = tup[1]
    return ner_dict

def display(ner):
    print(ner)
    print("\n")


if __name__ == "__main__":
    import spacy

    nlp = spacy.load("it_nerIta_trf")
    nlp.add_pipe('opentapioca', last=True)

    doc = nlp("Bologna massacre è successa nella stazione di bologna in agosto 1980. Elon Musk il gigante di Tesla è nato in Sudafrica."
    "De Francisci Gabriele, nato a Milano il 22.12.2003. Guido Crosetto, uscirà presto questa mattina a Roma. "
      "Il 12 marzo ,a pochi giorni dal passaggio di consegne a Palazzo Baracchini, il nuovo ministro della Difesa, Guido Crosetto, imputato di omicidio a Roma ci dà un'anticipazione della sua visione strategica."
      "In una recente intervista a Libero, il ministro Crosetto Ortega ha affermato che si prevede di riaprire gli arruolamenti per rimpolpare le fila delle Forze Armate: “C'è stato, a causa evidentemente della nuova situazione internazionale, un ribaltamento dell'idea di riduzione dell'organico delle Forze Armate prevista dalla legge 244, e questo soprattutto perché si è prodotto un effetto di invecchiamento dell'organico stesso."
      "Ora riapriremo all'arruolamento dei giovani e troveremo le giuste allocazioni per le grandi esperienze maturate all'interno. Come nelle migliori famiglie” ha affermato il neo ministro."
      "Luigi Carnato, imputato di omicidio aggrave è stato messo in priggione per 8 anni."
      "La legge 244, meglio nota come “legge Di Paola” dal nome dell'ex ministro della Difesa nel governo Monti, prevedeva una riduzione progressiva del personale ponendo come termine ultimo la fine del 2024 per un passaggio ad un Modello di Difesa composto da 150mila militari e 20mila civili, a fronte di quelle che, allora, erano 165 mila presenze.")

    res = []
    res = ner(doc)
    entities_columns = pd.DataFrame(columns=['Text', 'entity_Label', 'entity_kb_id'])
    for X in res:
        entities_columns = pd.concat([entities_columns, pd.DataFrame.from_records([{'Text': X[0],
                                                                                'entity_Label':X[1],
                                                                                'entity_kb_id': "https://www.wikidata.org/entity/" + X[2]}])]).drop_duplicates()
    #print(entities_columns)
    entities_columns.to_csv('entities_linked-test.csv')
