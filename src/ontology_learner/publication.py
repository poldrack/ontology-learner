from collections import defaultdict
from pathlib import Path
import json
import os

class Publication:
    def __init__(self, pmcid, datadir=None,
                 additional_sections=None):
        self.pmcid = pmcid
        self.data = None
        self.fulltext = None
        self.sections = {}
        if datadir is None:
            datadir = Path('../../data/json')
        self.datadir = datadir
        self.sections_to_include = [
            'TITLE', 'ABSTRACT', 'INTRO', 'METHODS', 'RESULTS', 
            'DISCUSS'] #, 'CONCL', 'FIG', 'TABLE']
        if additional_sections is not None:
            self.sections_to_include.extend(additional_sections)

    def load_json(self):
        fname = os.path.join(self.datadir, f'{self.pmcid}.json')
        with open(fname) as f:
            self.data = json.load(f)
        
    def parse_sections(self):
        if self.data is None:
            self.load_json()
        sections = {}
        for passage in self.data['documents'][0]['passages']:
            section_type = passage['infons']['section_type']
            if section_type not in sections:
                sections[section_type] = []
            sections[section_type].append(passage['text'] + '\n')
        for section, text in sections.items():
            self.sections[section] = ' '.join(text)
    
    def combine_text(self, sections_to_include=None):
        if sections_to_include is None:
            sections_to_include = self.sections_to_include
        combined_sections = [
            self.sections[section]
            for section in sections_to_include
            if section in self.sections and self.sections[section]
        ]
        return '\n'.join(combined_sections)



# Example usage:
# publication = ScientificPublication("Sample Title", ["Author1", "Author2"], "Sample Journal", 2023)
# publication.load_full_text_from_pdf("path_to_pdf_file.pdf")
# print(publication.full_text)