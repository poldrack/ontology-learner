import pytest
from .publication import Publication

def test_smoke():
    # Create a Publication instance
    pub = Publication('11557247', datadir='data/json')
    pub.load_json()
    pub.parse_sections()
    pub.combine_text()
    
    # Check if the full_text attribute is not empty
    assert pub.fulltext != ""
    

if __name__ == "__main__":
    pytest.main()