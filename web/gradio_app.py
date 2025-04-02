import os
import gradio as gr
import sys

# Add the whisper folder to path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "whisper"))

# Import infer from inference.py
from inference import infer

# Paths to test samples
REAL_SAMPLE = os.path.join("..", "whisper", "test_audio_real.mp3")
FAKE_SAMPLE = os.path.join("..", "whisper", "test_audio_fake.wav")

# Inference
def predict(audio_path):
    if not audio_path:
        return "<span style='color:red;'>Please upload or select an audio file.</span>"

    filename = os.path.basename(audio_path)
    result = infer(audio_path)
    color = "#4CAF50" if "real" in result.lower() else "#FF4C4C"  # Green or Red
    
    return f"<span style='color:{color}; font-size: 1.2rem; font-weight: bold;'>Prediction for <code>{filename}</code>: {result}</span>"

def set_sample(path):
    return path

def clear_all():
    return None, ""

with gr.Blocks() as detector:
    gr.Markdown("# 🕵️‍♂️ Deep Fake Audio Detector")
    gr.Markdown("Upload your own audio or try one of the sample clips. Click **Predict** to analyze.")

    # Shared audio player for upload/sample
    upload_box = gr.Audio(type="filepath", label="Selected Audio")

    with gr.Row():
        # Hidden paths to use as inputs
        real_path = gr.Textbox(value=REAL_SAMPLE, visible=False)
        fake_path = gr.Textbox(value=FAKE_SAMPLE, visible=False)
        real_btn = gr.Button("🎧 Try Sample Real")
        fake_btn = gr.Button("🎭 Try Sample Fake")

    with gr.Row():
        predict_btn = gr.Button("🧠 Predict", variant="primary")
        clear_btn = gr.Button("🧼 Clear", elem_classes=["custom-clear"])

    result_box = gr.HTML(label="Prediction")

    # Set sample into upload_box
    real_btn.click(fn=set_sample, inputs=[real_path], outputs=[upload_box])
    fake_btn.click(fn=set_sample, inputs=[fake_path], outputs=[upload_box])

    predict_btn.click(fn=predict, inputs=[upload_box], outputs=[result_box])

    clear_btn.click(fn=clear_all, inputs=[], outputs=[upload_box, result_box])

    gr.HTML("""
    <style>
        .custom-clear {
            background-color: #f5f5f5 !important;
            color: black !important;
            border: 1px solid #ccc;
        }
    </style>
    """)

if __name__ == "__main__":
    detector.launch(allowed_paths=["../whisper"])
