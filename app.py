import gradio as gr
from search import  answer
from Text_extractor import process_pdf
import os

with gr.Blocks() as demo:

    gr.Markdown("# 📄 RAG Chatbot")

    # PDF upload
    pdf = gr.File(
        label="Upload PDF",
        file_types=[".pdf"]
    )

    upload_button = gr.Button("Process PDF")
    status = gr.Textbox(label="Status")

    upload_button.click(
        fn=process_pdf,
        inputs=pdf,
        outputs=status
    )

    # Question
    question = gr.Textbox(
        label="Ask a question"
    )

    ask_button = gr.Button("Ask")

    response = gr.Textbox(
        label="Answer"
    )

    ask_button.click(
        fn=answer,
        inputs=question,
        outputs=response
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
