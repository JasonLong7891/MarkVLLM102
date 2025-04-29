from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import torch
from watermark.auto_watermark import AutoWatermarkForVLLM
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LogitsProcessorList
from visualize.visualizer import DiscreteVisualizer
from visualize.color_scheme import ColorSchemeForDiscreteVisualization
from visualize.font_settings import FontSettings
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
import base64
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model and components
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

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
    try:
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        transformers_config = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            vocab_size=config.vocab_size,
            device=device,
            max_new_tokens=256,
            max_length=256,
            do_sample=True,
            no_repeat_ngram_size=4
        )
        
        return (model, tokenizer), transformers_config
    except Exception as e:
        logger.error(f"Error initializing model {model_name}: {str(e)}")
        raise

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

def generate_text(model_tuple, prompt):
    """Generate text from prompt without watermark."""
    model, tokenizer = model_tuple
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            min_new_tokens=16,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_text_with_watermark(model_tuple, watermark, prompt):
    """Generate text with watermark."""
    model, tokenizer = model_tuple
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    
    # Generate text with watermark
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            min_new_tokens=16,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([watermark.watermark.logits_processor])
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        
        logger.info(f"Processing request with model: {model_name}, watermark: {watermark_method}")
        
        # Initialize components
        model_tuple, transformers_config = initialize_model(model_name)
        watermark = create_watermark(watermark_method, transformers_config)
        visualizer = create_visualizer()
        
        # Generate original text (without watermark)
        original_text = generate_text(model_tuple, prompt)
        if not original_text:
            logger.error("Failed to generate text")
            return jsonify({"error": "Failed to generate text"})
        
        # Generate watermarked text
        watermarked_text = generate_text_with_watermark(model_tuple, watermark, prompt)
        
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
        
        logger.info("Successfully processed request")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 10000))
    
    # Get debug mode from environment variable (default to False for production)
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port} with debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
