# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RandoMario is a Super Mario Bros. automation project built in Python that uses OpenAI Gym's `gym-super-mario-bros` environment. The project features an advanced bandit learning AI agent with statistical optimization, exponential decay, persistent storage, and Top-K replay diversity that plays Mario automatically while displaying both the game screen and a visual representation of the Nintendo Famicom controller inputs in real-time using Pygame.

## Core Architecture

### Core Files
- **Main Entry Point**: `main.py` - Contains the entire bandit learning application logic (~1000+ lines)
- **Test Implementation**: `test.py` - Alternative Go-Explore style Mario agent with UI integration (~600+ lines)
- **Optical Flow Implementation**: `optical_flow.py` - Go-Explore agent enhanced with real-time optical flow tracking and computer vision (~800+ lines)
- **Progress Dashboard**: `progress_viewer.py` - GUI dashboard for monitoring stage completion status (~350+ lines)
- The main codebase follows a monolithic structure with all functionality in single files

### Key Components in main.py:
- **Command Line Interface**: Stage selection via `--stage` or `-s` arguments (e.g., 1-1, 2-1, 4-1, 8-4)
- **Action Set Management**: Multiple predefined action configurations (RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, etc.) with position-based switching logic
- **Pygame UI System**: Dual-screen display showing game window and controller visualization
- **Bandit Learning Agent**: Statistically optimized action selection using UCB1 + ε-greedy algorithms with X-coordinate binning
- **Replay System**: Partial replay with X-coordinate based cutoff and adaptive exploration
- **Stall Detection**: Automatic boost mode activation when progress stagnates
- **Controller Visualization**: Real-time highlighting of pressed buttons on a Famicom controller image
- **Auto Image Download**: Automatically downloads controller image from external source if missing

### Core Function Architecture (main.py):
**Learning Functions:**
- `choose_action_with_bandit(bin_id, candidate_indices)`: UCB1 action selection with exploration
- `update_bandit_from_transition(prev_decision, curr_x, done, info, prev_x, is_stalled)`: Reward calculation with penalties
- `get_x_bin(x)`: Maps X-coordinates to learning bins
- `calculate_sequence_diversity(seq1, seq2)`: Edit distance for replay diversity

**Persistence Functions:**
- `save_stats(stage)` / `load_stats(stage)`: JSON-based cross-session learning
- `convert_numpy_types(obj)`: Handles numpy serialization for JSON compatibility

**Game Loop Functions:**
- `game_loop(env, stage)`: Main execution loop with episode management
- `update_memory_at_episode_end()`: Top-K sequence management with diversity filtering
- `setup_stage_strategy(stage)`: Dynamic action set configuration per stage

**UI Functions:**
- `draw_controller_state_ui()`: Real-time controller button visualization
- `draw_text_info()`: HUD overlay with learning statistics

### Action System Architecture
The codebase uses a sophisticated action switching system with statistical learning:

**Stage Strategy System:**
- `STAGE_STRATEGIES`: Dictionary containing stage-specific strategy configurations
- `X_THRESHOLDS_FOR_ACTION_SET_SWITCH`: Defines X-coordinate thresholds for action set changes
- `ALLOWED_ACTIONS_SUBSETS_BY_X`: Maps position ranges to specific action sets
- `ACTION_SET_NAMES_BY_X`: Human-readable names for each action set

**Bandit Learning System:**
- `X_BIN_SIZE`: Divides X-coordinates into bins for localized learning (default: 20px, range: 10-40)
- `bandit_stats`: Tracks success/failure statistics for each action in each X-bin
- `UCB_C`: Upper Confidence Bound exploration parameter (default: 1.2, range: 0.8-2.0)
- `EPSILON_GREEDY`: Random exploration probability (default: 0.10, range: 0.05-0.20)

**Adaptive Exploration:**
- `REPLAY_BACKOFF_X`: Early replay cutoff distance from best X (default: 120px, range: 80-160)
- `STALL_TIME_SEC`: Stagnation detection threshold (default: 2.0s, range: 1.5-3.0)
- `DEATH_PENALTY`: Penalty for death/falling actions (default: 20.0, range: 20-40)
- `BOOST_DECISIONS`: Aggressive action count during stall recovery (default: 8)

**Supported Stage Strategies:**
- **4-4**: 6-stage strategy with specialized actions for maze navigation
- **6-2**: 3-stage strategy with simple movement transitions
- **7-4**: 10-stage complex strategy with precise dash/jump combinations
- **default**: Basic RIGHT_ONLY strategy for all other stages

The system automatically selects the appropriate strategy based on the specified stage and optimizes action selection through statistical learning.

### Key Components in test.py:
- **Go-Explore Algorithm**: Cell-based archive system for systematic exploration
- **Cell Archive**: Discretizes game states into (x_bin, y_bin, status) cells with path storage
- **Return-then-Explore**: Two-phase approach - return to promising cells, then explore from there
- **Heuristic Action Sampling**: Probabilistic action selection with jump/dash/movement patterns
- **Archive Persistence**: Saves/loads exploration progress using pickle format
- **Real-time UI**: Same controller visualization system as main.py

**Archive System:**
- `Cell`: Dataclass storing cell_id, path sequence, max_x reached, visit count
- `_select_start_cell_path()`: Weighted selection of promising cells based on max_x and visit frequency
- `_maybe_add_or_update_cell()`: Updates archive with better paths or higher x-progress

**Exploration Heuristics:**
- `down_press_prob`: 2% chance to press down (for pipes/secret areas)
- `left_adjust_prob`: 5% chance for left movement (position adjustment)
- `jump_start_prob`: 12% chance to start jump sequence (6-14 frame holds)
- Main action: right dash for forward progress

### Key Components in progress_viewer.py:
- **Real-time Monitoring**: Automatic file watching with 2-second intervals to detect pkl changes
- **GUI Dashboard**: Tkinter-based 4×8 matrix showing all stages (1-1 through 8-4)
- **Status Visualization**: Color-coded progress indicators with episode tracking
  - ✅ Green: Stage cleared (shows first clear episode)
  - 🔄 Orange: Stage in progress (shows current episode)
  - ⚫ Gray: Stage not attempted
- **Dark Theme UI**: Modern design with Segoe UI fonts and professional color scheme
- **Error Handling**: Robust pickle loading with fallback mechanisms for corrupted files

**Technical Features:**
- `ProgressData`: Class managing individual stage data with modification tracking
- `ProgressViewerGUI`: Main GUI class with threading for file monitoring
- Automatic pkl/ directory scanning with multi-action-set support
- Background monitoring thread with graceful shutdown handling

### Key Components in optical_flow.py:
- **FlowBasedTracker**: Advanced computer vision system for real-time motion analysis
- **Camera Motion Estimation**: Robust background flow analysis with HUD exclusion and EMA filtering
- **Object Detection & Tracking**: Mario identification via residual flow analysis and connected components
- **Motion Quantification**: Real-time velocity and acceleration calculation in both screen and world coordinates
- **Visual Overlay System**: Real-time tracking visualization with bounding boxes and velocity vectors
- **Archive Integration**: Full Go-Explore compatibility with optical flow-enhanced decision making

**Optical Flow System Architecture:**
- `FlowBasedTracker`: Core tracking class using OpenCV's Farneback optical flow algorithm
- `_estimate_camera_flow()`: Median-based camera motion estimation with HUD masking
- `_segment_moving_objects()`: Residual flow analysis for independent object motion detection
- `_identify_mario()`: Mario-specific tracking with size/position heuristics and temporal consistency
- `scale_bbox_to_screen()`: Coordinate transformation for visualization overlay
- `_draw_optical_flow_overlay()`: Real-time visual feedback for tracked objects and motion vectors

**Motion Analysis Features:**
- **Background Subtraction**: Separates camera movement from object motion using residual flow
- **Adaptive Thresholding**: Dynamic motion detection based on flow magnitude distribution
- **Temporal Stability**: Exponential moving average filtering for smooth camera motion estimation
- **Mario Detection**: Multi-frame tracking with sprite size heuristics and nearest-neighbor association
- **Velocity Estimation**: Dual coordinate system (screen vs world) velocity calculation with acceleration derivatives
- **Multi-object Tracking**: Detection and tracking of up to 8 secondary moving objects (enemies, power-ups, etc.)

## Development Commands

### Setup and Installation
```bash
# Install dependencies using uv
uv sync

# Run the main application (default: stage 1-1)
uv run main.py

# Run with specific stage (uses specialized strategy if available)
uv run main.py --stage 7-4  # Uses 10-stage complex strategy
uv run main.py -s 4-4       # Uses 6-stage maze strategy
uv run main.py -s 6-2       # Uses 3-stage simple strategy
uv run main.py -s 1-1       # Uses default strategy

# Tune learning parameters for optimization
uv run main.py --stage 7-4 --x-bin-size 10 --death-penalty 40      # Fine-grained learning
uv run main.py --stage 4-4 --replay-backoff 160 --epsilon 0.15     # High diversity exploration
uv run main.py --stage 1-1 --stall-time 1.5 --ucb-c 1.8          # Aggressive stall detection

# Run the Go-Explore test implementation
uv run test.py --stage 1-1 --episodes 1000                            # Basic Go-Explore run (saves to pkl/)
uv run test.py --stage 4-2 --actions complex --archive pkl/4-2.pkl    # Complex actions with custom archive
uv run test.py --fps 120 --max-steps 8000                             # High FPS with extended episodes

# Run the Optical Flow enhanced implementation
uv run optical_flow.py --stage 1-1                                    # Basic optical flow tracking (saves to pkl/)
uv run optical_flow.py --stage 4-4 --actions complex                  # Complex actions with computer vision
uv run optical_flow.py --fps 120 --max-steps 8000                     # High FPS with extended motion analysis

# Use original random behavior (disable all learning)
uv run main.py --stage 1-1 --random                               # Pure random action selection

# Monitor progress across all stages
python progress_viewer.py                                         # Launch progress dashboard GUI

# Show help for command line options (includes parameter ranges)
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
- `opencv-python>=4.11.0.86`: Computer vision library used extensively in optical_flow.py for real-time motion analysis

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
Action behaviors are controlled through multiple sophisticated systems:

**Stage Configuration:**
- `STAGE_STRATEGIES` dictionary defines position-based action sets per stage
- `setup_stage_strategy()` function dynamically configures behavior
- Easy to add new stages by extending the configuration

**Learning Parameters:**
All bandit learning parameters are configurable via command line arguments:
- `--x-bin-size`: Binning granularity (10-40, smaller=local optimization, larger=generalization)
- `--replay-backoff`: Early exploration distance (80-160, larger=more diversity)
- `--stall-time`: Stagnation sensitivity (1.5-3.0 seconds)
- `--death-penalty`: Learning penalty for deaths (20-40, adjust by stage difficulty)
- `--epsilon`: Random exploration rate (0.05-0.20)
- `--ucb-c`: UCB exploration strength (0.8-2.0)

**Adaptive Systems:**
- `get_x_bin()`: Maps positions to learning bins
- `choose_action_with_bandit()`: Statistical action selection
- `update_bandit_from_transition()`: Enhanced reward-based learning updates with time penalty, backtrack detection, and stall penalty

### Advanced Learning Features (v2 Improvements)

**Exponential Decay System:**
- `DECAY = 0.99`: Applies exponential decay to bandit statistics for non-stationary adaptation
- Past statistics are weighted down to help escape local optima
- Enables better adaptation to environment changes

**Enhanced Reward System:**
- Time penalty: -0.01 per decision to favor faster progress
- Backtrack penalty: -2.0 when X position decreases
- Stall penalty: -1.0 during detected stagnation periods
- Dynamic reward calculation based on progress patterns

**Persistent Storage System:**
- `save_stats(stage)` and `load_stats(stage)`: JSON-based persistence per stage
- Automatic saving every 100 episodes and on program exit
- Cross-session learning continuity with stage-specific statistics

**Top-K Replay Diversity:**
- `TOP_K_SEQUENCES = 5`: Maintains multiple successful sequences instead of just one
- `calculate_sequence_diversity()`: Edit distance-based diversity scoring
- Minimum diversity threshold prevents overly similar sequences
- Random sequence selection (20% probability) from Top-K during replay
- ±20px randomization of replay cutoff positions for exploration variety

### Auto-Resource Management
The application automatically downloads required assets (controller image) from external sources if not present locally.

## File Structure
```
randomario/
├── fig/                    # Contains controller image assets
│   └── famicon01_01.png   # Nintendo Famicom controller image (auto-downloaded)
├── pkl/                   # Archive storage directory
│   ├── go_explore_archive_*.pkl     # Go-Explore agent archives (from test.py)
│   └── optical_flow_archive_*.pkl   # Optical flow enhanced archives (from optical_flow.py)
├── main.py                # Single main application file (~1100+ lines)
├── test.py               # Go-Explore implementation (~600+ lines)
├── optical_flow.py       # Optical flow enhanced Go-Explore (~800+ lines)
├── progress_viewer.py    # Progress monitoring GUI (~350+ lines)
├── pyproject.toml        # Project configuration and dependencies
├── uv.lock              # Dependency lock file
├── .gitignore           # Excludes auto-generated files
├── bandit_*.json        # Learning statistics (auto-generated, gitignored)
├── CLAUDE.md            # This development guide
└── README.md            # Project documentation (Japanese)
```

**Generated Files:**
- `bandit_{stage}.json`: Persistent learning statistics per stage containing:
  - `bandit`: X-bin action statistics with exponential decay weights
  - `sequences`: Top-K successful action sequences with diversity scores
  - Automatically saved every 100 episodes and on exit
  - Stage-specific format (e.g., `bandit_1_1.json`, `bandit_7_4.json`)
- `pkl/optical_flow_archive_{stage}_{actions}.pkl`: Optical flow enhanced archives containing:
  - `archive`: Go-Explore cell archive with path sequences
  - `best_overall_x`: Maximum X position reached
  - `episodes_done`: Total episodes completed
  - `first_clear_episode`: Episode number of first stage completion

## Development Notes

### Testing and Validation
```bash
# Syntax validation
uv run python -c "import main; print('Syntax check passed')"

# Quick functionality test (run for a few episodes)
timeout 30s uv run main.py --stage 1-1

# Performance parameter testing
uv run main.py --stage 1-1 --random  # Test legacy mode
```

No formal test framework is implemented. Testing is primarily done through runtime execution and parameter validation.

### Common Issues and Debugging

**JSON Serialization Errors:**
```bash
# If numpy types cause JSON errors, check convert_numpy_types() function
# Error: "Object of type int64 is not JSON serializable"
```

**Performance Tuning:**
```bash
# Monitor learning progress with debug output
uv run main.py --stage 1-1 2>&1 | grep "DEBUG\|Top-K\|sequence"

# Test different learning parameters
uv run main.py --stage 4-4 --x-bin-size 15 --ucb-c 1.5  # Fine-tune difficult stages
uv run main.py --stage 1-1 --epsilon 0.05 --stall-time 1.5  # Conservative exploration
```

**Data Reset:**
```bash
# Clear learning statistics for fresh start
rm bandit_*.json

# Verify clean state
ls bandit_*.json  # Should show "No such file or directory"
```

### AI Enhancement Features
The project implements advanced learning capabilities:

**Current AI Systems:**
- **Bandit Learning**: UCB1 + ε-greedy for position-aware action optimization
- **Adaptive Replay**: X-coordinate based early cutoff from successful sequences
- **Stall Detection**: Automatic boost mode when progress stagnates
- **Reward Learning**: Death penalties and progress rewards shape behavior
- **Optical Flow Tracking**: Real-time computer vision for motion analysis and object tracking
- **Go-Explore Integration**: Cell-based exploration with visual feedback enhancement

**Computer Vision Capabilities:**
- **Real-time Motion Analysis**: Farneback optical flow for camera and object motion estimation
- **Background Subtraction**: Separates screen scrolling from object movement
- **Multi-object Tracking**: Simultaneous tracking of Mario and up to 8 other moving objects
- **Velocity Quantification**: Dual-coordinate velocity and acceleration calculations
- **Visual Debugging**: Real-time overlay visualization for tracking validation

**Future Enhancement Areas:**
Based on dependencies, the project is designed to support:
- Advanced computer vision algorithms (DIS/RAFT optical flow, feature-based tracking)
- Data visualization and analysis (matplotlib available)  
- More sophisticated reinforcement learning algorithms
- LinUCB/Thompson sampling contextual bandits
- Macro-action (option) hierarchical learning

### Localization
The project includes extensive Japanese documentation and comments, indicating a bilingual codebase.