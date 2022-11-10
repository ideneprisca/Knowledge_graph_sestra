#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clausie as a spacy library
"""
import numpy as np
import pandas as pd
import spacy
import logging
import typing

from spacy.tokens import Span, Doc
from spacy.matcher import Matcher
from DocParsimg import listdirs, data_cleaning

logging.basicConfig(level=logging.INFO)

# DO NOT SET MANUALLY
MOD_CONSERVATIVE = False

Doc.set_extension("clauses", default=[], force=True)
Span.set_extension("clauses", default=[], force=True)

dictionary = {
    "non_ext_copular": """morire caminare""".split(),
    "ext_copular": """agire
residere
nominare
rifiutare
uccidere 
vivere
nascere
apparire
essere
avvenire
divenire
venire
Vienire 
procedere
finire
Ottenere
andare
crescere
cadere
sentire
mantenere
partire
guardare
provare
rimanere
sembrare
odorare
suonare
restare
gustare
girare
presentarsi
caricare
vivere
venire
andare
Alzarsi
mentire
amare
fare

try""".split(),
    "complex_transitive": """
    rifiutare
portare
prendere
Guidare
Ottenere
mantenere
giacere
mettere
sedere
mostrare
Alzarsi
scivolare
take""".split(),
    "adverbs_ignore": """so
allora
così
perché
come
con
mai
""".split(),
    "adverbs_include": """
appena
raramente
recentemente""".split(),
}


class Clause:
    def __init__(
            self,
            subject: typing.Optional[Span] = None,
            verb: typing.Optional[Span] = None,
            indirect_object: typing.Optional[Span] = None,
            direct_object: typing.Optional[Span] = None,
            complement: typing.Optional[Span] = None,
            adverbials: typing.List[Span] = None,
    ):
        """


        Parameters
        ----------
        subject : Span
            Subject.
        verb : Span
            Verb.
        indirect_object : Span, optional
            Indirect object, The default is None.
        direct_object : Span, optional
            Direct object. The default is None.
        complement : Span, optional
            Complement. The default is None.
        adverbials : list, optional
            List of adverbials. The default is [].

        Returns
        -------
        None.

        """
        if adverbials is None:
            adverbials = []

        self.subject = subject
        self.verb = verb
        self.indirect_object = indirect_object
        self.direct_object = direct_object
        self.complement = complement
        self.adverbials = adverbials

        self.doc = self.subject.doc

        self.type = self._get_clause_type()

    def _get_clause_type(self):
        has_verb = self.verb is not None
        has_complement = self.complement is not None
        has_adverbial = len(self.adverbials) > 0
        has_ext_copular_verb = (
                has_verb and self.verb.root.lemma_ in dictionary["ext_copular"]
        )
        has_non_ext_copular_verb = (
                has_verb and self.verb.root.lemma_ in dictionary["non_ext_copular"]
        )
        conservative = MOD_CONSERVATIVE
        has_direct_object = self.direct_object is not None
        has_indirect_object = self.indirect_object is not None
        has_object = has_direct_object or has_indirect_object
        complex_transitive = (
                has_verb and self.verb.root.lemma_ in dictionary["complex_transitive"]
        )

        clause_type = "undefined"

        if not has_verb:
            clause_type = "SVC"
            return clause_type

        if has_object:
            if has_direct_object and has_indirect_object:
                clause_type = "SVOO"
            elif has_complement:
                clause_type = "SVOC"
            elif not has_adverbial or not has_direct_object:
                clause_type = "SVO"
            elif complex_transitive or conservative:
                clause_type = "SVOA"
            else:
                clause_type = "SVO"
        else:
            if has_complement:
                clause_type = "SVC"
            elif not has_adverbial or has_non_ext_copular_verb:
                clause_type = "SV"
            elif has_ext_copular_verb or conservative:
                clause_type = "SVA"
            else:
                clause_type = "SV"

        return clause_type

    def __repr__(self):
        return "<{}, {}, {}, {}, {}, {}, {}>".format(
            self.type,
            self.subject,
            self.verb,
            self.indirect_object,
            self.direct_object,
            self.complement,
            self.adverbials,
        )

    def to_propositions(self):
        propositions = []
        subjects = extract_ccs_from_token_at_root(self.subject)
        direct_objects = extract_ccs_from_token_at_root(self.direct_object)
        indirect_objects = extract_ccs_from_token_at_root(self.indirect_object)
        complements = extract_ccs_from_token_at_root(self.complement)
        verbs = [self.verb] if self.verb else []

        for subj in subjects:
            ###print("my subject",subj,subjects)
            if complements and not verbs:
                ###print("it is complement but not a verbs",complements,verbs)
                for c in complements:
                    ###print("i k complement",c)
                    # perche nella proposition si aggiunge il "è"
                    propositions.append((subj, c))
                    #print("************************i k complement",subj)
                propositions.append((subj, "") + tuple(complements))
            for verb in verbs:
                # print("tutti i verbi",verbs)
                prop = [subj, verb]
                ###print("type of ....",self.type)
                '''if self.type in ["SV", "SVA"]:
                   if self.adverbials:
                        ###print("is  adverbial")
                        for a in self.adverbials:
                            propositions.append(tuple(prop + [a]))
                        propositions.append(tuple(prop + self.adverbials))
                    else:
                        ###print("not adverbial")
                        propositions.append(tuple(prop))'''
                if self.type == "SVOO":
                    for iobj in indirect_objects:
                        for dobj in direct_objects:
                            propositions.append((subj, verb, iobj))
                            propositions.append((subj, verb, dobj))
                elif self.type == "SVO":
                    for obj in direct_objects + indirect_objects:
                        propositions.append((subj, verb, obj))
                        for a in self.adverbials:
                            propositions.append((subj, verb, obj, a))

                elif self.type == "SVOC":
                    for obj in indirect_objects + direct_objects:
                        if complements:
                            #for c in complements:
                                #propositions.append(tuple(prop + [obj, c]))
                            propositions.append(tuple(prop + [obj]))
                elif self.type == "SVOA":
                    for obj in direct_objects:
                        if self.adverbials:
                            #for a in self.adverbials:
                                #propositions.append(tuple(prop + [obj, a]))
                            propositions.append(tuple(prop + [obj]))

                '''elif self.type == "SVC":
                    if complements:
                        for c in complements:
                            propositions.append(tuple(prop + [c]))
                        propositions.append(tuple(prop + complements))'''
        # Remove doubles
        propositions = list(set(propositions))
        # print("partial porpo",propositions)
        return propositions





#verb detect function lunch with clean/noise text
def _get_verb_matches(clean_text, span):
    if clean_text: #clean text
        verb_matcher = Matcher(span.vocab)
        # print("verb_matcher",)
        verb_matcher.add("Auxiliary verb phrase aux-verb", [[{"POS": "AUX"}, {"POS": "VERB"}, {"POS": "ADP"}]])
        verb_matcher.add("Auxiliary verb phrase", [[{"POS": "AUX"}]])
        verb_matcher.add("Auxiliary verb phrase aux-verb-", [[{"POS": "AUX"}, {"POS": "VERB"}]])
        verb_matcher.add("Verb phrase", [[{"POS": "VERB"}]] )
        result = verb_matcher(span)
        return result
    elif not clean_text:  #noise text
        verb_matcher = Matcher(span.vocab)
        # print("verb_matcher",)
        verb_matcher.add("Auxiliary verb phrase aux-verb", [[{"POS": "AUX"}, {"POS": "VERB"}]])
        result = verb_matcher(span)
        return result
    #return result


def _get_verb_chunks(clean_text, span):
    matches = _get_verb_matches(clean_text, span)

    # Filter matches (e.g. do not have both "has won" and "won" in verbs)
    verb_chunks = []
    for match in [span[start:end] for _, start, end in matches]:
        if match.root not in [vp.root for vp in verb_chunks]:
            verb_chunks.append(match)
    return verb_chunks


def _get_subject(verb):
    root = verb.root
    ###print("verb************",verb)
    ###print("depandancy of verb", root, root.dep_,root.head)
    ###print("verb root",root)
    while root:
        # Can we find subject at current level?
        for c in root.children:
            #print("root verb c", c,c.dep_)
            if c.dep_ in ["nsubj", "nsubj:pass","flat:name"]:
                subject = extract_span_from_entity(c)
                # print("real sunìbject",subject)
                return subject

        # ... otherwise recurse up one level
        if (root.dep_ in ["conj", "cc", "advcl", "acl", "ccomp"]
                and root != root.head):
            root = root.head
            ###print("final rooot",root)
        else:
            root = None
    return None


def _find_matching_child(root, allowed_types):
    for c in root.children:
        if c.dep_ in allowed_types:
            return extract_span_from_entity(c)
    return None


def extract_clauses(clean_text, span):
    # print("span of clause:",span,"++++fine span")
    clauses = []

    verb_chunks = _get_verb_chunks(clean_text, span)
    # print("verb_chunks:",verb_chunks)
    for verb in verb_chunks:

        subject = _get_subject(verb)

        #print("real subject",subject)
        if not subject:
            continue

        # Check if there are phrases of the form, "AE, a scientist of ..."
        # If so, add a new clause of the form:
        # <AE, is, a scientist>
        for c in subject.root.children:
            if c.dep_ == "detposs":
                complement = extract_span_from_entity(c)
                clause = Clause(subject=subject, complement=complement)
                clauses.append(clause)

        indirect_object = _find_matching_child(verb.root, ["obl", "pobj", "obj"])
        direct_object = _find_matching_child(verb.root, ["dobj","expl" ,"obj"])
        complement = _find_matching_child(
            verb.root, ["ccomp", "acomp", "xcomp", "attr", "case"]
        )
        adverbials = [
            extract_span_from_entity(c)
            for c in verb.root.children
            if c.dep_ in ("advmod","advmod", "agent","case","det","nummod")
        ]

        clause = Clause(
            subject=subject,
            verb=verb,
            indirect_object=indirect_object,
            direct_object=direct_object,
            complement=complement, # write complment in clause's set
            adverbials=adverbials,
        )
        clauses.append(clause)
        #print("clause of span", clause)
    return clauses

#launch when text is clean
@spacy.Language.component('claucy')
def extract_clauses_doc(doc):
    for sent in doc.sents:
            clauses = extract_clauses(clean_text=True, span=sent)
            sent._.clauses = clauses
            doc._.clauses += clauses
    return doc

#launch when text have noise
@spacy.Language.component('extract')
def extract(doc):
    for sent in doc.sents:
            clauses = extract_clauses(clean_text=True, span=sent)
            sent._.clauses = clauses
            doc._.clauses += clauses
    return doc


def add_to_pipe(nlp):
    nlp.add_pipe('claucy')
    nlp.add_pipe('extract')


def extract_span_from_entity(token):
    ent_subtree = sorted([c for c in token.subtree], key=lambda x: x.i)
    return Span(token.doc, start=ent_subtree[0].i, end=ent_subtree[-1].i + 1)


def extract_span_from_entity_no_cc(token):
    ent_subtree = sorted(
        [token] + [c for c in token.children if c.dep_ not in ["cc", "conj"]],
        key=lambda x: x.i,
    )
    return Span(token.doc, start=ent_subtree[0].i, end=ent_subtree[-1].i + 1)


def extract_ccs_from_entity(token):
    entities = [extract_span_from_entity_no_cc(token)]
    for c in token.children:
        if c.dep_ in ["conj", "cc"]:
            entities += extract_ccs_from_entity(c)
    return entities


def extract_ccs_from_token_at_root(span):
    # print("span",span)
    if span is None:
        return []
    else:
        result = extract_ccs_from_token(span.root)
        # print("span result ", result)

        return result


def extract_ccs_from_token(token):
    if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
        children = sorted(
            [token]
            + [
                c
                for c in token.children
                if c.dep_ in ["advmod", "amod", "det","case", "poss","nummod","flat:name"]
            ],
            key=lambda x: x.i,
        )
        entities = [Span(token.doc, start=children[0].i, end=children[-1].i + 1)]
    else:
        entities = [Span(token.doc, start=token.i, end=token.i + 1)]
    for c in token.children:
        if c.dep_ == "conj":
            entities += extract_ccs_from_token(c)
    return entities



if __name__ == "__main__":
    import spacy
    import rdflib

    from testNer import ner_to_dict, ner

    nlp = spacy.load("it_core_news_sm")

    rootdir = 'C:/Users/user/Desktop/thesis/sestra/bo0144'
    nlp = spacy.load("it_core_news_sm")
    add_to_pipe(nlp)


    pipeline_with_clean_text = "claucy"
    pipeline_with_noise = "extract"

    nlp.disable_pipe(pipeline_with_clean_text)

    doc = nlp("Bologna massacre è successa nella stazione di bologna in agosto 1980. Elon Musk il gigante di Tesla è nato in Sudafrica."
        "De Francisci Gabriele, nato a Milano il 22.12.2003. Guido Crosetto, uscirà presto questa mattina a Roma. "
      "Il 12 marzo ,a pochi giorni dal passaggio di consegne a Palazzo Baracchini, il nuovo ministro della Difesa, Guido Crosetto, imputato di omicidio a Roma ci dà un'anticipazione della sua visione strategica."
      "In una recente intervista a Libero, il ministro Crosetto Ortega ha affermato che si prevede di riaprire gli arruolamenti per rimpolpare le fila delle Forze Armate: “C'è stato, a causa evidentemente della nuova situazione internazionale, un ribaltamento dell'idea di riduzione dell'organico delle Forze Armate prevista dalla legge 244, e questo soprattutto perché si è prodotto un effetto di invecchiamento dell'organico stesso."
      "Ora riapriremo all'arruolamento dei giovani e troveremo le giuste allocazioni per le grandi esperienze maturate all'interno. Come nelle migliori famiglie” ha affermato il neo ministro."
      "Luigi Carnato, imputato di omicidio aggrave è stato messo in priggione per 8 anni."
      "Mario Rossi è andato a Parigi in dicembre 1980 e ha avuto un contratto di lavoro con la Sacmi Coop .  Luigi Carnota è nato a Bologna il 22/09/2022."
     "Un gatto, sentendo che gli uccelli in una certa voliera stavano male si traveste da medico e, preso il suo bastone e la sua  borsa di strumenti che erano diventati la sua professione, andò a far loro visita."
      "La legge 244, meglio nota come “legge Di Paola” dal nome dell'ex ministro della Difesa nel governo Monti, prevedeva una riduzione progressiva del personale ponendo come termine ultimo la fine del 2024 per un passaggio ad un Modello di Difesa composto da 150mila militari e 20mila civili, a fronte di quelle che, allora, erano 165 mila presenze.")


    # write all propositions (subject, verb, object) in csv  file
    Triplet_columns = pd.DataFrame(columns=['subject', 'relation', 'object'])
    for clause in doc._.clauses:
        res = clause.to_propositions()
        if len(res) > 0:
            for tuple_i in res:
                Triplet_columns = pd.concat(
                    [Triplet_columns, pd.DataFrame.from_records([{'subject': tuple_i[0],
                                                                  'relation': tuple_i[1],
                                                                  'object': tuple_i[2],
                                                                  }])])
            Triplet_columns.to_csv('./data/Output/Ner/tri_relation.csv')
########################################################################################################
#test Named entity and triple matching for create rdflib
    res = ner(doc)
    nerdict = ner_to_dict(res)
    dict_entity = ner_to_dict(res)
    entity_set = set(dict_entity.keys())

    final_triples = []
    for row, col in Triplet_columns.iterrows():
        col['subject'] = str(col['subject']).strip()
        # check if there is  Named Entity in subject sentence fragment
        found_entity = False
        for named_entity in entity_set:
            if named_entity in col['subject']:
                col['subject'] = named_entity
                found_entity = True
        if found_entity:
            added = False
            entity2_sent = col['object']
            for entity in entity_set:
                    final_triples.append(('Node', col['subject'], col['relation'], 'Node', col['object']))

    DfTriple_matched_and_processed= pd.DataFrame(final_triples, columns=['Type1','Entity1','Relationship','Type2', 'Entity2']).drop_duplicates()
    DfTriple_matched_and_processed.to_csv('./data/Output/Ner/Triple_matched_and_processed.csv')
#print(final_df)

#create  file (entity, entuty_label, link URI)


#################################################################################################
#knowledge graph construction e generazione rdf files
    Triple_matched_and_processed = DfTriple_matched_and_processed.values.tolist()
    #print(Triple_matched_and_processed)

    Triple_matched_and_processed_graph = rdflib.Graph()
    
    for triple in Triple_matched_and_processed:
        #print(triple)
        Triple_matched_and_processed_graph.add((
            rdflib.Literal(triple[1], datatype=rdflib.namespace.XSD.string),
            rdflib.Literal(triple[2], datatype=rdflib.namespace.XSD.string),
            rdflib.Literal(triple[4], datatype=rdflib.namespace.XSD.string)
        ))
    for s, p, o in Triple_matched_and_processed_graph:
        print(s, '->', p, '->', o)






