import requests
import os
import json


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

    def download_pdfs(self, ids, download_dir):
        fetch_url = f"{self.base_url}efetch.fcgi"
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        
        for pmcid in ids:
            params = {
                "db": "pmc",
                "id": pmcid,
                "retmode": "xml",
                "email": self.email
            }
            response = requests.get(fetch_url, params=params)
            response.raise_for_status()
            xml_data = response.text

            # Extract PDF URL from the XML data
            pdf_url = self.extract_pdf_url(xml_data)
            if pdf_url:
                pdf_response = requests.get(pdf_url)
                pdf_response.raise_for_status()
                pdf_path = os.path.join(download_dir, f"{pmcid}.pdf")
                with open(pdf_path, "wb") as pdf_file:
                    pdf_file.write(pdf_response.content)
                print(f"Downloaded {pdf_path}")
            else:
                print(f"No PDF found for PMCID {pmcid}")

    def extract_pdf_url(self, xml_data):
        # This is a placeholder implementation. You need to parse the XML data to extract the PDF URL.
        # You can use libraries like xml.etree.ElementTree or lxml to parse the XML.
        # For simplicity, let's assume the PDF URL is directly available in the XML data.
        # Replace this with actual XML parsing logic.
        return "http://example.com/sample.pdf"


if __name__ == "__main__":
    email = "poldrack@stanford.edu"
    pmc_search = PubMedCentralSearch(email)
    query = 'brain AND open access[filter] AND "fMRI" OR "functional MRI" OR "functional magnetic resonance imaging"'
    ids = pmc_search.search(query)
    download_dir = "../../data/json"
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    pmc_search.download_json(ids, download_dir)