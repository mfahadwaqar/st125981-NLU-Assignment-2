"""
Harry Potter Language Model Web Application

This Flask web application provides an interactive interface for generating
Harry Potter-style text using a trained LSTM language model.

Author: st125981
Course: Artificial Intelligence - Natural Language Understanding
Assignment: 2
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import pickle
import os
import math
import re
from collections import Counter

# Initialize Flask app
app = Flask(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BasicEnglishTokenizer:
    """Simple tokenizer that mimics torchtext's basic_english tokenizer"""
    def __call__(self, text):
        # Convert to lowercase and split by whitespace/punctuation
        text = text.lower()
        # Add spaces around punctuation
        text = re.sub(r"([.!?,;:])", r" \1 ", text)
        # Split on whitespace
        tokens = text.split()
        return tokens


class Vocab:
    """Simple vocabulary class that mimics torchtext.vocab"""
    def __init__(self, tokens, min_freq=1, specials=None):
        self.specials = specials or []
        self.min_freq = min_freq
        
        # Count token frequencies
        counter = Counter(tokens)
        
        # Build vocabulary: special tokens first, then frequent tokens
        self.itos = self.specials.copy()
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.itos:
                self.itos.append(token)
        
        # Create reverse mapping
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.default_index = self.stoi.get('<unk>', 0)
    
    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)
    
    def get_itos(self):
        return self.itos
    
    def set_default_index(self, index):
        self.default_index = index


class LSTMLanguageModel(nn.Module):
    """
    LSTM-based Language Model for text generation.
    Same architecture as training notebook.
    """
    
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, 
                           dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)
        
    def init_hidden(self, batch_size, device):
        """Initialize hidden state and cell state"""
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
    
    def forward(self, src, hidden):
        """Forward pass through the network"""
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden


def load_model():
    """
    Load the trained model, vocabulary, and configuration.
    
    Returns:
        model: Trained LSTM model
        vocab: Vocabulary object
        tokenizer: Tokenizer function
    """
    # Load configuration
    with open('models/config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    # Load vocabulary
    with open('models/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    # Initialize model with saved configuration
    model = LSTMLanguageModel(
        vocab_size=config['vocab_size'],
        emb_dim=config['emb_dim'],
        hid_dim=config['hid_dim'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load('models/best_harry_potter_lm.pt', 
                                     map_location=device))
    model.eval()
    
    # Initialize tokenizer
    tokenizer = BasicEnglishTokenizer()
    
    return model, vocab, tokenizer


def generate_text(prompt, max_length=100, temperature=0.7, model=None, 
                 vocab=None, tokenizer=None):
    """
    Generate text continuation from a prompt.
    
    Args:
        prompt: Input text to continue from
        max_length: Maximum number of tokens to generate
        temperature: Controls randomness (0.5-1.5 recommended)
        model: Trained language model
        vocab: Vocabulary object
        tokenizer: Tokenizer function
    
    Returns:
        Generated text as string
    """
    if model is None or vocab is None or tokenizer is None:
        return "Model not loaded"
    
    model.eval()
    
    # Tokenize and encode the prompt
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    
    # Initialize hidden state
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    
    with torch.no_grad():
        for i in range(max_length):
            # Convert current sequence to tensor
            src = torch.LongTensor([indices]).to(device)
            
            # Get model prediction
            prediction, hidden = model(src, hidden)
            
            # Apply temperature and get probability distribution
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Skip <unk> tokens
            while next_token == vocab['<unk>']:
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Stop if we generate <eos>
            if next_token == vocab['<eos>']:
                break
            
            # Add predicted token to sequence
            indices.append(next_token)
    
    # Convert indices back to tokens
    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    
    return ' '.join(tokens)


# Load model at startup
print("Loading model...")
model, vocab, tokenizer = load_model()
print("Model loaded successfully!")


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """
    API endpoint for text generation.
    
    Expects JSON with:
        - prompt: Input text
        - max_length: Maximum tokens to generate (optional)
        - temperature: Sampling temperature (optional)
    
    Returns:
        JSON with generated text
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt', 'Harry Potter is')
        max_length = int(data.get('max_length', 100))
        temperature = float(data.get('temperature', 0.7))
        
        # Validate inputs
        if not prompt.strip():
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        if max_length < 10 or max_length > 500:
            return jsonify({'error': 'Max length must be between 10 and 500'}), 400
        
        if temperature < 0.1 or temperature > 2.0:
            return jsonify({'error': 'Temperature must be between 0.1 and 2.0'}), 400
        
        # Generate text
        generated_text = generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            model=model,
            vocab=vocab,
            tokenizer=tokenizer
        )
        
        return jsonify({
            'success': True,
            'prompt': prompt,
            'generated_text': generated_text,
            'max_length': max_length,
            'temperature': temperature
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    # Run the Flask app
    print("\n" + "=" * 50)
    print("Harry Potter Language Model Web App")
    print("=" * 50)
    print(f"Server running on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)