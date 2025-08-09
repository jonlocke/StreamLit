#!/bin/bash
sudo apt -yinstall python3.10-venv python3-pip
sudo ollama pull deepseek-r1:latest # Default 7B model
ollama pull mxbai-embed-large  # Embeddings model
# set up project
git clone https://github.com/paquino11/chatpdf-rag-deepseek-r1.git
cd chatpdf-rag-deepseek-r1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PATH="$HOME/.local/bin:$PATH"
streamlit run app.py
