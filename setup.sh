#!/bin/bash

# Quick Setup Script for Ceramic Armor ML API (macOS/Linux)
# This script automates the setup process for macOS and Linux users

echo ""
echo "========================================"
echo " Ceramic Armor ML API - Quick Setup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.10 or 3.11 and try again"
    echo ""
    echo "On macOS: brew install python@3.11"
    echo "On Ubuntu: sudo apt update && sudo apt install python3.11 python3.11-venv"
    exit 1
fi

echo "✅ Python found"
python3 --version

echo ""
echo "🔧 Running automated setup..."
echo ""

# Run the Python setup script
python3 setup.py

echo ""
echo "📋 Quick Start Commands:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the application:"
echo "   python start_server.py"
echo ""
echo "3. Open in browser:"
echo "   http://localhost:8000"
echo ""

# Make the script executable
chmod +x setup.sh