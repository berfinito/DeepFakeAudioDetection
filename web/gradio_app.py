import gradio as gr

def predict(audio):
    return "Prediction: Real (placeholder)"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Deep Fake Audio Detector",
    description="Upload an audio file to check if it's real or fake.",
)

if __name__ == "__main__":
    interface.launch()
