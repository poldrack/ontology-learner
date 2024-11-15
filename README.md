This project will use LLMs to learn an ontology from papers downloaded from pubmed central.

The steps will be:

- search pubmed central using the specified search terms
- use JSON or papers matching the search term
    - obtained from full BioC-PMC download - https://ftp.ncbi.nlm.nih.gov/pub/wilbur/BioC-PMC/README.txt
- load each PDF and identify relevant features from the text: 
    - psychological constructs
    - psychological tasks
    - brain regions
    - tables with activation data

