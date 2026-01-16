import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import os
from PyPDF2 import PdfReader
from utils.clean_text import clean_text
from utils.extract_text_from_file import extract_text_from_file


# --- Model Loading ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "arabert_basev2")

print(f"Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("Warning: Model path not found locally. Falling back to HuggingFace if needed.")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
    
    # helper pipeline
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error loading model: {e}")
    qa_pipeline = None


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

