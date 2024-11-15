import pytest
from .publication import Publication

def test_load_full_text_from_pdf():
    # Create a Publication instance
    publication = Publication("Sample Title", ["Author1", "Author2"], "Sample Journal", 2023)
    
    # Load the full text from the PDF file
    pdf_path = "data/pdf/zns1643.pdf"
    publication.load_full_text_from_pdf(pdf_path)
    
    # Check if the full_text attribute is not empty
    assert publication.full_text != ""
    

if __name__ == "__main__":
    pytest.main()