This project will use LLMs to learn an ontology from papers downloaded from pubmed central.

The steps will be:

- search pubmed central using the specified search terms
- download PDFs for papers matching the search term
- load each PDF and identify relevant features from the text:
    - psychological constructs
    - psychological tasks
    - brain regions
    - tables with activation data

