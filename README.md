# LLM Watermarking Web Interface

A web interface for applying and visualizing watermarks on LLM-generated text.

## Features

- Support for multiple LLM models (GPT-2, GPT-2 Medium, etc.)
- Multiple watermarking methods (Unigram, UPV, KGW)
- Interactive web interface
- Real-time text generation and watermarking
- Visualization of watermarked text

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/markllm.git
cd markllm
```

2. Create and activate a virtual environment:
```bash
python -m venv markllm_venv
source markllm_venv/bin/activate  # For Linux/Mac
# OR
.\markllm_venv\Scripts\activate  # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Select a model, watermark method, and enter your prompt.

## Project Structure

```
markllm/
├── app.py                 # Flask application
├── requirements.txt       # Project dependencies
├── templates/
│   └── index.html        # Web interface template
└── README.md             # Project documentation
```

## Requirements

- Python 3.8+
- CUDA-capable GPU
- See requirements.txt for Python dependencies

## License

[Your chosen license]
