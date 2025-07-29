# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- `uv run python app.py` - Start the development server on port 5001
- `uv sync` - Install/update dependencies and create virtual environment

### Package Management
This project uses `uv` for fast Python package management. Dependencies are defined in `pyproject.toml`.

## Architecture Overview

### Core Components

**Flask Application (`app.py`)**
- Single-file Flask application serving the Rock Paper Scissors game
- Runs on port 5001 by default
- Uses session-based storage with filesystem predictor persistence

**ML Prediction System**
- `ImprovedMLPredictor` class implements the core AI logic
- Uses n-gram pattern recognition (1-7 move sequences) to predict player moves
- Employs confidence-based decision making with automatic fallback to randomness
- Stores predictor state in temporary files using pickle serialization
- Session cleanup occurs probabilistically (5% chance per request) for predictors older than 24 hours

**Frontend Architecture**
- Traditional server-rendered HTML with AJAX interactions
- Uses Chart.js for move frequency visualization and D3.js for pattern heatmaps
- JavaScript handles game interactions and real-time visualization updates
- Static assets organized in `/static/` with separate CSS and JS directories

### Key Technical Details

**Session Management**
- Each player gets a unique session ID stored in Flask sessions
- Predictor models are persisted to filesystem in `/tmp/rps_predictors/`
- Session data includes scores, round numbers, and predictor references

**AI Strategy**
- Follows "The Bitter Lesson" - pure pattern learning without human heuristics
- Variable n-gram weighting with preference for longer patterns when sufficient data exists
- Confidence threshold system (default 0.55) determines when to use patterns vs. randomness
- Real-time pattern analysis with entropy calculations and predictability metrics

**Data Flow**
- User moves → Update predictor model → Generate counter-prediction → Return game state + visualizations
- All game state changes happen via POST requests to `/play` endpoint
- Visualization data (frequencies, transitions, insights) computed server-side and sent to frontend

### Project Structure
```
rock-paper-scissors/
├── app.py                    # Main Flask application with ML predictor
├── templates/index.html      # Game interface with visualizations
├── static/
│   ├── css/style.css        # Game styling
│   ├── js/script.js         # Frontend game logic and Chart.js integration
│   └── favicon.ico          # Site icon
└── pyproject.toml           # uv package configuration
```