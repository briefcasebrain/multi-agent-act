#!/bin/bash

# Setup script for virtual environment

echo "Setting up virtual environment for multi-agent-collab-learning..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install torch
pip install matplotlib
pip install networkx
pip install pytest
pip install pytest-cov

# Install package in development mode
pip install -e .

echo "Virtual environment setup complete!"
echo "To activate: source venv/bin/activate"
echo "To run tests: pytest"