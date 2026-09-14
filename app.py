"""
Gradio Web Interface for live sentiment and recommendation analysis.
"""

import gradio as gr
from src.predictor import SentimentPredictor

predictor = SentimentPredictor()

def analyze_review(review_text: str):
    if not review_text or not review_text.strip():
        return "Please input a valid Persian customer review.", "0.0%"

    prediction, confidence = predictor.predict(review_text)
    return prediction, f"{confidence * 100:.1f}%"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # 🛒 Persian E-Commerce Review & Recommendation Classifier
        ### Lightweight Natural Language Processing Pipeline for Consumer Sentiment Analysis
        This service automatically classifies Persian customer reviews into three actionable categories:
        **Recommended**, **Not Recommended**, or **Neutral / Undecided**.
        """
    )

    with gr.Row():
        with gr.Column():
            review_input = gr.Textbox(
                label="Persian Customer Review",
                placeholder="Paste or write a customer review in Persian...",
                lines=4
            )
            submit_btn = gr.Button("Analyze Review", variant="primary")

        with gr.Column():
            prediction_output = gr.Label(label="Predicted Purchase Intent")
            confidence_output = gr.Textbox(label="Confidence Score")

    gr.Examples(
        examples=[
            ["کیفیت ساخت فراتر از انتظارم بود، خرید این محصول را کاملاً پیشنهاد می‌کنم."],
            ["کیفیت جنس افتضاح بود و با تصاویر سایت مطابقت نداشت. مرجوع کردم."],
            ["نسبت به قیمتش در تخفیف بد نیست، کارایی معمولی دارد."]
        ],
        inputs=review_input
    )

    submit_btn.click(
        fn=analyze_review,
        inputs=review_input,
        outputs=[prediction_output, confidence_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
