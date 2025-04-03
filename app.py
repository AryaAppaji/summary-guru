import torch, gradio as gr

from transformers import pipeline

text_summary = pipeline(task="summarization", model=r"ai_models\models--sshleifer--distilbart-cnn-12-6\snapshots\a4f8f3ea906ed274767e9906dbaede7531d660ff")

def summarize_text(input):
    output = text_summary(input)
    return output[0]["summary_text"]

ui_demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(label="Enter your input text", lines=20),
    outputs=gr.Textbox(label="Summarized Text", lines=5),
    title="Summary Guru",
    description="This app will summarize the given input text",
    article="Created by Arya as a part of AI app developement course"
)

ui_demo.launch()