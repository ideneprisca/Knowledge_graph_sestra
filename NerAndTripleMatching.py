'''   
        import spacy
        from testNer import ner_dict

        nlp = spacy.load("it_core_news_sm")
        nlp.add_pipe("entityfishing", last=True)


        rootdir = 'C:/Users/user/Desktop/thesis/sestra/bo0144'
        nlp = spacy.load("it_core_news_sm")
        add_to_pipe(nlp)
        #nlp.add_pipe("entityLinker", last=True)

        pipeline_with_clean_text = "claucy"
        pipeline_with_noise = "extract"

        nlp.disable_pipe(pipeline_with_noise)
        pageContent = listdirs(rootdir)
        allContent = data_cleaning(pageContent)
         #write all propositions (subject, verb, object) in csv  file
        csv_input = []
        Triplet_columns = pd.DataFrame(columns=['Subject', 'Predicate', 'Object'])
        for k in range(len(allContent)):
            doc = nlp(allContent[k])
            for clause in doc._.clauses:
                res = clause.to_propositions()
                if len(res)>0:
                    for tuple_i in res:
                        if len(tuple_i)== 3:
                               Triplet_columns = pd.concat(
                                   [Triplet_columns, pd.DataFrame.from_records([{'Subject': tuple_i[0],
                                                                                 'Predicate': tuple_i[1],
                                                                                 'Object':tuple_i[2],
                                                                                 }])])
        Triplet_columns.to_csv('./data/Output/Ner/tri_relation.csv')
 '''









