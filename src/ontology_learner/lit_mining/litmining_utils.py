from ontology_learner.publication import Publication

def get_fulltext_from_pmcid_json(pmcid, datadir):

    pub = Publication(pmcid, datadir=datadir)
    pub.load_json()
    pub.parse_sections()    
    return pub.combine_text()
