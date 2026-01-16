import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import os
from PyPDF2 import PdfReader
from utils.clean_text import clean_text



# --- Model Loading ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "arabert_basev2")

print(f"Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("Warning: Model path not found locally. Falling back to HuggingFace if needed.")
    # Fallback or error handling could go here, but per plan we use the local path.
    # Note: If local path is missing, transformers might try to look it up online if it looks like a repo ID, 
    # but "models/arabert_basev2" is definitely a path.

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
    
    # helper pipeline
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error loading model: {e}")
    qa_pipeline = None

# --- File Processing ---
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

# --- Main QA Function ---
def answer_question(file_obj, question):
    if not qa_pipeline:
        return "Error: Model not loaded."
    
    if file_obj is None:
        return "Please upload a file."
    
    if not question.strip():
        return "Please ask a question."

    context = extract_text_from_file(file_obj)
    
    if not context or "Error" in context:
        return "Could not extract content from file or file is empty."
        
    cleaned_question = clean_text(question)
    
    # The pipeline handles truncation/stride effectively for basic usage
    try:
        result = qa_pipeline(question=cleaned_question, context=context)
        return result['answer']
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# --- Gradio Interface ---
# Custom CSS for a more "Pro" look if needed, but Soft theme is usually good.
custom_css = """
#component-0 {border-radius: 10px; border: 1px solid #e0e0e0;}
"""

with gr.Interface(
    fn=answer_question,
    inputs=[
        gr.File(label="Upload Context (PDF or TXT)", file_types=[".pdf", ".txt"]),
        gr.Textbox(label="Your Question", placeholder="Ask something about the document...", lines=2)
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Pro Arabic QA System",
    description="Upload a document and ask questions using the fine-tuned AraBERTv2 model.",
) as demo:
    pass

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=custom_css)

