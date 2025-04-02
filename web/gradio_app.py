import os
import gradio as gr
import sys

# Add the whisper folder to path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "whisper"))

# Import infer from inference.py
from inference import infer

def predict(audio_path):
    if audio_path is None:
        return "No audio uploaded."
    
    try:
        prediction = infer(audio_path)
        return f"Prediction: {prediction}"
    except Exception as e:
        return f"Error: {e}"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Deep Fake Audio Detector",
    description="Upload an audio file to check if it's real or fake.",
)

if __name__ == "__main__":
    interface.launch()
