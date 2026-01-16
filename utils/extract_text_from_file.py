from PyPDF2 import PdfReader
from utils.clean_text import clean_text

def extract_text_from_file(file_obj):
    if file_obj is None:
        return ""
    
    content = ""
    try:
        file_path = file_obj.name
        if file_path.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n"
        elif file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            return "Unsupported file format. Please upload PDF or TXT."
            
        return clean_text(content)
    except Exception as e:
        return f"Error reading file: {str(e)}"
