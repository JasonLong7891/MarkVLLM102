from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from vllm import LLM, SamplingParams
import torch
from watermark.auto_watermark import AutoWatermarkForVLLM
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from visualize.visualizer import DiscreteVisualizer
from visualize.color_scheme import ColorSchemeForDiscreteVisualization
from visualize.font_settings import FontSettings
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
import base64
from io import BytesIO

app = Flask(__name__)

# Initialize model and components
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Available models and watermark methods
AVAILABLE_MODELS = {
    "gpt2": "GPT-2",
    "gpt2-medium": "GPT-2 Medium",
    "gpt2-large": "GPT-2 Large",
    "gpt2-xl": "GPT-2 XL"
}

AVAILABLE_WATERMARKS = ["Unigram", "UPV", "KGW"]

def initialize_model(model_name):
    """Initialize the model and related components."""
    model = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=256,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        dtype="auto"
    )
    
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_hf = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    transformers_config = TransformersConfig(
        model=model_hf,
        tokenizer=tokenizer,
        vocab_size=config.vocab_size,
        device=device,
        max_new_tokens=256,
        max_length=256,
        do_sample=True,
        no_repeat_ngram_size=4
    )
    
    return model, transformers_config

def create_watermark(algorithm_name, transformers_config):
    """Create watermark instance."""
    return AutoWatermarkForVLLM(
        algorithm_name=algorithm_name,
        algorithm_config=f'config/{algorithm_name}.json',
        transformers_config=transformers_config
    )

def create_visualizer():
    """Create visualizer instance."""
    return DiscreteVisualizer(
        color_scheme=ColorSchemeForDiscreteVisualization(),
        font_settings=FontSettings(),
        page_layout_settings=PageLayoutSettings(),
        legend_settings=DiscreteLegendSettings()
    )

def generate_text(model, prompt):
    """Generate text from prompt."""
    outputs = model.generate(
        prompts=[prompt],
        sampling_params=SamplingParams(
            n=1,
            temperature=1.0,
            seed=42,
            max_tokens=256,
            min_tokens=16
        )
    )
    
    if outputs and hasattr(outputs[0], 'outputs') and len(outputs[0].outputs) > 0:
        return outputs[0].outputs[0].text
    return None

def create_visualization(visualizer, watermark, text):
    """Create visualization of watermarked text."""
    img = visualizer.visualize(
        data=watermark.get_data_for_visualization(text=text),
        show_text=True,
        visualize_weight=True,
        display_legend=True
    )
    
    # Convert PIL Image to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', 
                         models=AVAILABLE_MODELS,
                         watermarks=AVAILABLE_WATERMARKS)

@app.route('/process', methods=['POST'])
def process():
    """Process the user's request."""
    try:
        # Get form data
        model_name = request.form['model']
        watermark_method = request.form['watermark']
        prompt = request.form['prompt']
        
        # Initialize components
        model, transformers_config = initialize_model(model_name)
        watermark = create_watermark(watermark_method, transformers_config)
        visualizer = create_visualizer()
        
        # Generate original text
        original_text = generate_text(model, prompt)
        if not original_text:
            return jsonify({"error": "Failed to generate text"})
        
        # Apply watermark
        watermarked_text = watermark.apply_watermark(original_text)
        
        # Create visualizations
        original_viz = create_visualization(visualizer, watermark, original_text)
        watermarked_viz = create_visualization(visualizer, watermark, watermarked_text)
        
        # Prepare response
        response = {
            "original_text": original_text,
            "watermarked_text": watermarked_text,
            "original_viz": original_viz,
            "watermarked_viz": watermarked_viz
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 