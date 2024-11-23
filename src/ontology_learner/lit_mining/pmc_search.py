import requests
import os
import json
import shutil


class PubMedCentralSearch:
    def __init__(self, email):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = email

    def search(self, query):
        search_url = f"{self.base_url}esearch.fcgi"
        params = {
            "db": "pmc",
            "term": query,
            "retmax": 200000,
            "retmode": "json",
            "email": self.email
        }
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        result = data["esearchresult"]["idlist"]
        print(f'Found {len(result)} results for query: {query}')
        return result

    # bioc_dir contains complete download of open access portion of pmc
    # we want to copy all of our target articles into a separate directory
    # for ease of use
    # also, the bioc json files were misnamed as ".xml" so we fix that
    def copy_from_bioc_download(self, ids, bioc_dir, target_dir):
        missing_pmids = []
        copied_files = []
        for pmcid in ids:
            infile = os.path.join(bioc_dir, f'PMC{pmcid}.xml')
            outfile = os.path.join(target_dir, f'{pmcid}.json')
            if not os.path.exists(infile):
                missing_pmids.append(pmcid)
                continue
            if os.path.exists(outfile):
                continue
            shutil.copyfile(infile, outfile)
            copied_files.append(pmcid)
        print(f"copied {len(copied_files)} files")
        print(f"missing {len(missing_pmids)} files")

    # deprecated - lots of missing files
    def download_json(self, ids, download_dir, ntries=4):
        for pmcid in ids:
            json_path = os.path.join(download_dir, f"{pmcid}.json")
            if os.path.exists(json_path):
                continue
            print(f'Downloading {pmcid}')
            fetch_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC{pmcid}/unicode" 
            try_num = 0
            while try_num < ntries:
                try:
                    response = requests.get(fetch_url)
                    break
                except Exception as e:
                    print(e)
                    try_num += 1
            if try_num >= ntries:
                print(f"failed to retrieve {pmcid}")
                continue
            response.raise_for_status()
            try:
                json_data = response.json()
            except json.decoder.JSONDecodeError as e:
                print(response.text)
                continue
            with open(json_path, "w") as json_file:
                json_file.write(json.dumps(json_data[0]))



if __name__ == "__main__":
    email = "poldrack@stanford.edu"
    pmc_search = PubMedCentralSearch(email)
    query = 'brain AND open access[filter] AND "fMRI" OR "functional MRI" OR "functional magnetic resonance imaging"'
    ids = pmc_search.search(query)
    target_dir = "../../data/json"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    bioc_dir = "../../data/BioC_expanded"
    pmc_search.copy_from_bioc_download(ids, bioc_dir, target_dir)