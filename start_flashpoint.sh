#!/bin/bash
# FLASHPOINT V2 - Streaming State Machine - Quick Start

echo "============================================================"
echo "⚡ FLASHPOINT V2 - STREAMING STATE MACHINE"
echo "============================================================"
echo ""
echo "🚀 Features:"
echo "  • Token-level streaming updates"
echo "  • Speculative execution (parallel branch evaluation)"
echo "  • Sub-10ms cycle latency target"
echo "  • Cascade failure injection"
echo "  • Partial JSON parsing"
echo ""

# Check Python version
echo "📋 Step 1: Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✓ Found: $(python3 --version)"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "✓ Found: $(python --version)"
else
    echo "✗ Python not found! Please install Python 3.7+"
    exit 1
fi

echo ""
echo "📦 Step 2: Installing dependencies..."
echo "Running: pip install websockets openai"

$PYTHON_CMD -m pip install websockets openai --break-system-packages 2>/dev/null || \
$PYTHON_CMD -m pip install websockets openai --user 2>/dev/null || \
pip install websockets openai --break-system-packages 2>/dev/null || \
pip install websockets openai --user

echo ""
echo "🔑 Step 3: Setting up API key..."
echo ""
echo "IMPORTANT: You need your Cerebras API key!"
echo "Get it from: https://cloud.cerebras.ai/"
echo ""
read -p "Enter your Cerebras API key: " API_KEY

if [ -z "$API_KEY" ]; then
    echo "✗ No API key provided!"
    exit 1
fi

export CEREBRAS_API_KEY="$API_KEY"

echo ""
echo "============================================================"
echo "✓ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "📝 NEXT STEPS:"
echo ""
echo "1. Start the backend server (this will run now):"
echo "   $PYTHON_CMD flashpoint_engine.py"
echo ""
echo "2. Open flashpoint_ui.html in your browser"
echo ""
echo "3. Watch for:"
echo "   • Streaming hypothesis tokens appearing in real-time"
echo "   • Latency comparator showing microseconds vs milliseconds"
echo "   • Severity dropping from 9 → 2 in 10 cycles"
echo "   • Cascade failures being handled"
echo ""
echo "============================================================"
echo ""
echo "Press Enter to start the backend server now, or Ctrl+C to exit"
read -r

echo ""
echo "🚀 Starting FLASHPOINT V2 backend server..."
echo ""
echo "💡 TIP: Watch the terminal for cycle timing metrics!"
echo ""

$PYTHON_CMD flashpoint_engine.py
