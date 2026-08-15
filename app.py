import os

print("=== APP STARTED ===")
print("PORT =", os.environ.get("PORT"))

import gradio as gr

from search import answer
from Text_extractor import process_pdf

print("=== IMPORTS COMPLETED ===")


with gr.Blocks(title="RAG Chatbot") as demo:

    gr.Markdown("# 📄 RAG Chatbot")

    gr.Markdown(
        "Upload a PDF, process it, and then ask questions about its content."
    )

    # ==============================
    # PDF Upload
    # ==============================

    pdf = gr.File(
        label="Upload PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    upload_button = gr.Button("Process PDF")

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    upload_button.click(
        fn=process_pdf,
        inputs=pdf,
        outputs=status
    )

    # ==============================
    # Question
    # ==============================

    question = gr.Textbox(
        label="Ask a question",
        placeholder="Ask something about your PDF..."
    )

    ask_button = gr.Button("Ask")

    response = gr.Textbox(
        label="Answer",
        interactive=False
    )

    ask_button.click(
        fn=answer,
        inputs=question,
        outputs=response
    )


print("=== GRADIO UI CREATED ===")


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    print("=== STARTING GRADIO ===")
    print(f"Server will run on 0.0.0.0:{port}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
