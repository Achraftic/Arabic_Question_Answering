import gradio as gr
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import os
from utils.clean_text import clean_text
from utils.extract_text_from_file import extract_text_from_file


# --- Model Configuration ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
AVAILABLE_MODELS = {
    "AraBERT Base v2": "arabert_basev2",
    "AraBERT Large": "arabert_large",
    "AraELECTRA Base": "araELECTRA-base",
}

# Global state
current_model_name = None
qa_pipeline = None


def load_model(model_display_name):
    """Loads the model if it's not already loaded."""
    global current_model_name, qa_pipeline

    if model_display_name == current_model_name and qa_pipeline is not None:
        return qa_pipeline

    folder_name = AVAILABLE_MODELS.get(model_display_name)
    if not folder_name:
        return None

    model_path = os.path.join(MODELS_DIR, folder_name)
    print(f"Loading model from: {model_path}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        current_model_name = model_display_name
        print(f"Successfully loaded {model_display_name}")
        return qa_pipeline
    except Exception as e:
        print(f"Error loading model {model_display_name}: {e}")
        return None


# Pre-load the default model
print("Initializing default model...")
load_model("AraBERT Base v2")


# --- Main QA Function ---
def answer_question(model_choice, file_obj, text_context, question):
    pipeline_instance = load_model(model_choice)

    if not pipeline_instance:
        return "Error: Model could not be loaded."

    if not question.strip():
        return "Please ask a question."

    context = ""
    if file_obj is not None:
        context = extract_text_from_file(file_obj)
    elif text_context and text_context.strip():
        context = clean_text(text_context)

    if not context:
        return "Please provide a context (upload a file or paste text)."

    if "Error" in context:
        return context

    cleaned_question = clean_text(question)

    # The pipeline handles truncation/stride effectively for basic usage
    try:
        result = pipeline_instance(question=cleaned_question, context=context)
        return result["answer"]
    except Exception as e:
        return f"Error generating answer: {str(e)}"


custom_css = """
#component-0 {border-radius: 10px; border: 1px solid #e0e0e0;}
"""

with gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Dropdown(
            choices=list(AVAILABLE_MODELS.keys()),
            value="AraBERT Base v2",
            label="Choose Model",
        ),
        gr.File(label="Upload Context (PDF or TXT)", file_types=[".pdf", ".txt"]),
        gr.Textbox(
            label="Or Paste Context Here",
            placeholder="Paste your arabic text here...",
            lines=5,
        ),
        gr.Textbox(
            label="Your Question",
            placeholder="Ask something about the document...",
            lines=2,
        ),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Pro Arabic QA System",
    description="Upload a document OR paste text, choose a model, then ask questions.",
) as demo:
    pass

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=custom_css, share=True)
