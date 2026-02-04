# FlashPoint
A real-time AI control system that uses LLMs not as chat interfaces, but as control policies for autonomous incident response. Streaming token-level decisions enable sub-second action cycles.

## What You Need
- Python 3.7+ installed
- Chrome or Firefox (other browsers may fail WebSockets)
- All files in the same folder:
  ```
  flashpoint_engine.py
  flashpoint_ui.html
  start_flashpoint.sh (Mac/Linux)
  start_flashpoint.bat (Windows)
  ```
- Internet connection (for Cerebras API)
- **Cerebras API key** 

## One-Command Setup

### Windows:
1. Download all files
2. Double-click `start_flashpoint.bat`
   
   The script will:
   - Check Python installation
   - Install dependencies (websockets, openai)
   - Prompt for your Cerebras API key
   - Start the backend server

3. Open `flashpoint_ui.html` in your browser

### Mac/Linux:
1. Download all files
2. Make the startup script executable:
   ```bash
   chmod +x start_flashpoint.sh
   ```
3. Run:
   ```bash
   ./start_flashpoint.sh
   ```
   
   The script will:
   - Check Python installation
   - Install dependencies
   - Prompt for your Cerebras API key
   - Start the backend server

4. Open `flashpoint_ui.html` in your browser

## Manual Method (Optional)

If you prefer to do it manually:

1. Install dependencies:
   ```bash
   pip install websockets openai --break-system-packages
   ```

2. Set your Cerebras API key:
   ```bash
   # Mac/Linux
   export CEREBRAS_API_KEY='your-api-key-here'
   
   # Windows CMD
   set CEREBRAS_API_KEY=your-api-key-here
   
   # Windows PowerShell
   $env:CEREBRAS_API_KEY='your-api-key-here'
   ```

3. Start the backend server:
   ```bash
   python flashpoint_engine.py
   ```
   
   You should see:
   ```
   ======================================================================
   ⚡ FLASHPOINT - STREAMING AI CONTROL SYSTEM
   ======================================================================
   
   🚀 Architecture:
     • LLM-as-control-policy (not chat interface)
     • Token-level streaming (act before completion)
     • Speculative action evaluation (parallel scoring)
     • High-frequency state machine (~400ms per AI decision)
     • Cascade failure simulation
   
   🌐 WebSocket: ws://localhost:8765
   
   Waiting for connections...
   ```

4. Open the UI:
   - Double-click `flashpoint_ui.html` in file browser, OR
   - Drag file into browser window

## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|-------------|------|
| "CEREBRAS_API_KEY not set" | Missing API key | Run setup script again OR set manually (see step 2) |
| RED "Disconnected" status | Backend not running | Check terminal, restart `start_flashpoint` |
| Port 8765 in use | Another process | Kill process or edit port in both `.py` and `.html` |
| ModuleNotFoundError | Missing dependencies | `pip install websockets openai` |
| API authentication failed | Wrong/expired key | Get new key from https://cloud.cerebras.ai/ |
| Slow cycles (>5s each) | Network latency | Check connection; Cerebras API may be slow |
| No streaming tokens visible | WebSocket buffer issue | Refresh browser (F5) |

## Quick Test (Optional)

Verify WebSocket connection:
```python
import asyncio
import websockets

async def test_connection():
    async with websockets.connect('ws://localhost:8765') as ws:
        message = await ws.recv()
        print("Received:", message)

asyncio.run(test_connection())
```

If it prints JSON, WebSocket is working.

## Demo Flow

1. **Show connection**: GREEN "Connected" dot in top status bar
2. **Incident injection**: Watch severity spike to SEV 7-9
3. **Streaming tokens**: Highlight hypothesis appearing token-by-token in real-time
4. **Speed comparison**: Point out Cerebras ~400ms vs Legacy GPU ~3500ms
5. **Cascade propagation**: Auth → Database → Dependent services
6. **Severity reduction**: 9 → 2 in ~10 cycles (under 2 seconds total)
7. **Cycle latency**: Show microsecond breakdown (Python overhead vs LLM inference)
8. **Repeat**: New incident every ~8 seconds

## What to Watch For

- **Token streaming**: Partial AI responses appearing before completion
- **Latency breakdown**: State machine overhead (μs) vs LLM inference (ms)
- **Speculative execution**: Multiple branches evaluated in parallel
- **Cascade failures**: Dependencies failing in realistic patterns
- **Speed advantage**: 7-10x faster than traditional GPU inference

## Key Metrics

- **Cycle latency**: ~400ms (mostly Cerebras API inference)
- **State machine overhead**: <10ms (Python execution)
- **Severity reduction**: 7-9 points in 10 cycles
- **Time to resolution**: <2 seconds (vs 35+ on legacy GPU)

---

**Get your free Cerebras API key**: https://cloud.cerebras.ai/

**Need help?** Check terminal output for errors or browser console (F12) for WebSocket issues.
