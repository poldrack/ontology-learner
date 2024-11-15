import fitz  # PyMuPDF

class Publication:
    def __init__(self, title, authors, journal, year):
        self.title = title
        self.authors = authors
        self.journal = journal
        self.year = year
        self.full_text = ""

    def load_full_text_from_pdf(self, pdf_path):
        try:
            document = fitz.open(pdf_path)
            text = ""
            for page_num in range(document.page_count):
                page = document.load_page(page_num)
                text += page.get_text()
            self.full_text = text
        except Exception as e:
            print(f"An error occurred while loading the PDF: {e}")

# Example usage:
# publication = ScientificPublication("Sample Title", ["Author1", "Author2"], "Sample Journal", 2023)
# publication.load_full_text_from_pdf("path_to_pdf_file.pdf")
# print(publication.full_text)