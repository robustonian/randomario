# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RandoMario is a Super Mario Bros. automation project built in Python that uses OpenAI Gym's `gym-super-mario-bros` environment. The project features a random action agent that plays Mario automatically while displaying both the game screen and a visual representation of the Nintendo Famicom controller inputs in real-time using Pygame.

## Core Architecture

### Single-File Design
- **Main Entry Point**: `main.py` - Contains the entire application logic in one file (~1000+ lines)
- The codebase follows a monolithic structure where all functionality is contained in the main script

### Key Components in main.py:
- **Command Line Interface**: Stage selection via `--stage` or `-s` arguments (e.g., 1-1, 2-1, 4-1, 8-4)
- **Action Set Management**: Multiple predefined action configurations (RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, etc.) with position-based switching logic
- **Pygame UI System**: Dual-screen display showing game window and controller visualization
- **Random Agent**: Selects actions randomly from configured action sets based on Mario's X position
- **Controller Visualization**: Real-time highlighting of pressed buttons on a Famicom controller image
- **Auto Image Download**: Automatically downloads controller image from external source if missing

### Action System Architecture
The codebase uses a sophisticated action switching system:
- `X_THRESHOLDS_FOR_ACTION_SET_SWITCH`: Defines X-coordinate thresholds for action set changes
- `ALLOWED_ACTIONS_SUBSETS_BY_X`: Maps position ranges to specific action sets
- `ACTION_SET_NAMES_BY_X`: Human-readable names for each action set

Different levels have commented-out configurations (4-4, 6-2, 7-4) showing the system's flexibility.

## Development Commands

### Setup and Installation
```bash
# Install dependencies using uv
uv sync

# Run the main application (default: stage 1-1)
uv run main.py

# Run with specific stage
uv run main.py --stage 2-1
uv run main.py -s 4-1

# Show help for command line options
uv run main.py --help
```

### Runtime Controls
- **R key**: Reset current episode and start new one
- **ESC key**: Exit the program

## Dependencies and Environment

### Key Libraries (from pyproject.toml):
- `gym-super-mario-bros>=7.4.0`: Super Mario Bros. Gym environment
- `pygame>=2.6.1`: Game display and UI rendering
- `matplotlib>=3.7.5`: Currently unused but available for future data visualization
- `opencv-python>=4.11.0.86`: Currently unused but available for future computer vision features

### Python Version
- Requires Python 3.8+
- Uses `uv` for dependency management (uv.lock file present)

## Code Patterns

### Global State Management
The code uses extensive global variables for state management:
- Screen and display objects (`screen`, `clock`, fonts)
- Controller image and geometry data
- Game state variables (positions, action sets, feedback mechanisms)

### Configuration-Driven Approach
Action behaviors are controlled through configuration arrays rather than hardcoded logic, making it easy to create level-specific strategies by modifying the threshold and action set arrays.

### Auto-Resource Management
The application automatically downloads required assets (controller image) from external sources if not present locally.

## File Structure
```
randomario/
├── fig/                    # Contains controller image assets
│   └── famicon01_01.png   # Nintendo Famicom controller image
├── main.py                # Single main application file
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock               # Dependency lock file
└── README.md             # Project documentation (Japanese)
```

## Development Notes

### Testing
No test files or testing framework is currently implemented in this project.

### Future Enhancement Areas
Based on commented code and dependencies, the project is designed to support:
- Computer vision-based agents (opencv-python available)
- Data visualization and analysis (matplotlib available)
- More sophisticated AI agents beyond random selection

### Localization
The project includes extensive Japanese documentation and comments, indicating a bilingual codebase.