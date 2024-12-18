from dataclasses import dataclass
from pathlib import Path
import json
from collections import defaultdict
@dataclass
class PaperResults:
    results_dir: Path
    output_dir: Path
    
    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.constructs = []
        self.tasks = []
        self.conditions = {}
        self.contrasts = {}
        self.brain_regions = []
        self.results_files = []
        self.pmcid = []
        self.paper_results = {}
        self.task_to_pmcid = defaultdict(list)
        self.construct_to_pmcid = defaultdict(list)
        self.paper_results_serialized = {}
        
    # get results files
    def get_results_files(self):
        self.results_files = list(self.results_dir.glob('*.json'))
        self.results_files.sort()
        print(f'found {len(self.results_files)} results files')

    # load data files
    def parse_results_files(self):

        for filename  in self.results_files:
            self.pmcid = filename.stem
            with open(filename, 'r') as f:
                paper_results = json.load(f)
            self.paper_results[self.pmcid] = paper_results
            self.paper_results_serialized[self.pmcid] = json.dumps(paper_results)
            if 'construct' in paper_results:
                self.constructs.extend(paper_results['construct'])
                for construct in paper_results['construct']:
                    self.construct_to_pmcid[construct].append(self.pmcid)
            if 'task' in paper_results:
                self.tasks.extend(paper_results['task'])
                for task in paper_results['task']:
                    self.task_to_pmcid[task].append(self.pmcid)
            # these are dicts
            if 'condition' in paper_results:
                for task, value in paper_results['condition'].items():
                    if len(value) == 0:
                        continue
                    if task in self.conditions:
                        self.conditions[task].extend(value)
                    else:
                        self.conditions[task] = value
            if 'contrast' in paper_results:
                for task, value in paper_results['contrast'].items():
                    if len(value) == 0:
                        continue
                    if task in self.contrasts:
                        self.contrasts[task].extend(value)
                    else:
                        self.contrasts[task] = value
            if 'brain_region' in paper_results:
                self.brain_regions.extend(paper_results['brain_region'])
        # remove duplicates
        self.constructs = list(set(self.constructs))
        self.tasks = list(set(self.tasks))
        self.brain_regions = list(set(self.brain_regions))

