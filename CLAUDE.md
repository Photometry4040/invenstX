# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InvenstX is a comprehensive stock trading system that combines technical analysis, pattern recognition, backtesting, and reinforcement learning-based trading strategies. The system features both a Streamlit-based UI (`main.py` - full dashboard) and a simpler UI (`app.py` - basic analysis).

## Running the Application

```bash
# Main application with full dashboard (recommended)
streamlit run main.py

# Alternative simpler application (basic stock analysis)
streamlit run app.py

# Activate virtual environment first if not already active
source .venv/bin/activate  # macOS/Linux
```

## Testing

```bash
# Run unit tests
python -m unittest discover -s tests/unit

# Run integration tests
python -m unittest discover -s tests/integration

# Run specific test files
python -m unittest tests/test_rsi.py
python -m unittest tests/test_macd.py
```

## Python Environment

- Python 3.8 (virtual environment in `.venv/`)
- Key dependencies: streamlit, yfinance, tensorflow, stable-baselines3, pytorch
- Install dependencies: `pip install -r requirements.txt`

## Architecture

### Core Components

**1. Entry Points**
- `main.py`: Full Streamlit dashboard with 5 modes (stock analysis, RL trading, portfolio, performance monitoring, multi-agent system)
- `app.py`: Simplified Streamlit UI focused on basic stock analysis and technical indicators
- `config.py`: Centralized configuration for model paths, data settings, technical indicators, and training hyperparameters

**2. Data Layer** (`data/`)
- `fetch_data.py`: Yahoo Finance API integration for stock data retrieval
- `data_loader.py`: Data preprocessing and loading utilities
- Caching via `utils/cache_manager.py` for performance optimization

**3. Technical Indicators** (`indicators/`)
- `rsi.py`, `macd.py`: Basic technical indicators (RSI, MACD)
- `technical_indicators.py`: Comprehensive technical indicator calculations
- `patterns/`: Pattern detection modules
  - `pattern_analyzer.py`: Main pattern analysis orchestrator
  - `bollinger_bands.py`, `double_bottom.py`, `double_top.py`, `head_and_shoulders.py`: Specific pattern detectors
  - `candlestick_patterns.py`: Candlestick pattern recognition

**4. Reinforcement Learning** (`reinforcement_learning/`)
- `environment.py`: `StockTradingEnvironment` class - custom gym environment for RL agents with support for transaction costs and holding incentives
- `model.py`: DQN neural network implementation with batch normalization and dropout, uses MPS (Apple Silicon GPU) when available
- `train.py`: Training loop for RL agents with episode management
- `evaluate.py`: Model evaluation and performance metrics calculation
- `utils.py`: Helper utilities for RL components

**5. RL Models** (`rl_models.py`)
- `StockTradingEnv`: Alternative trading environment with configurable features and reward scaling
- `DQNAgent`, `A2CAgent`, `PPOAgent`: Different RL agent implementations
- Support for multiple investment styles (long-term, swing trade, default)

**6. Visualization** (`charts/`)
- `stock_charts.py`: Basic stock price charts
- `technical_charts.py`: Charts for technical indicators
- `pattern_charts.py`: Pattern visualization
- `volume_charts.py`: Volume analysis charts
- Uses Plotly for interactive charts

**7. Models Storage** (`models/`)
- Trained RL models saved as: `[TICKER]_[investment_style]_model.pth`
- Examples: `AAPL_long_term_model.pth`, `TSLA_swing_trade_model.pth`
- Best models saved separately: `best_model.pth`, `tsla_best_model.pth`

### Investment Styles (RL Training)

The system supports three distinct trading strategies configured in `environment.py`:

1. **Long Term**: γ=0.99, 100-day reward window, high transaction cost (0.002), holding incentive (0.001)
2. **Swing Trade**: γ=0.95, 20-day reward window, medium transaction cost (0.001), holding incentive (0.0005)
3. **Default**: γ=0.95, 5-day reward window, low transaction cost (0.0005), no holding incentive

These styles affect agent behavior through reward functions and encourage different trading frequencies.

### Key Data Flow

1. **Stock Analysis**: User input (ticker, date range) → `fetch_data.py` → Technical indicators calculation → Visualization via `charts/`
2. **Pattern Scanning**: Stock data → `PatternAnalyzer` → Pattern detectors → Signal generation → Display results
3. **RL Training**: Historical data → `StockTradingEnvironment` → DQN Agent → Training loop → Save model to `models/`
4. **RL Evaluation**: Test data + trained model → `evaluate_agent()` → Performance metrics (returns, Sharpe ratio, trade count)
5. **Backtesting**: Strategy rules → Historical data → Performance calculation → Visualization

## Important Implementation Notes

### Device Configuration (Apple Silicon)
- PyTorch models automatically use MPS (Metal Performance Shaders) when available: `torch.device("mps" if torch.backends.mps.is_available() else "cpu")`
- Check `model.py:12` for device configuration

### Session State Management (Streamlit)
- Critical state variables stored in `st.session_state`: ticker, start_date, end_date, dashboard_mode
- Always check for state existence before accessing: `st.session_state.get('key', default_value)`
- Mode switching handled via sidebar radio in `main.py:87-94`

### Model Training & Evaluation
- Training uses epsilon-greedy exploration (starts at 1.0, decays to 0.05 by default)
- Evaluation mode MUST set epsilon to 0 to prevent random actions during testing
- Models saved with investment style suffix to avoid conflicts between strategies
- Reward function in `environment.py` incorporates: price changes, transaction costs, holding incentives

### Pattern Detection
- Patterns are detected using sliding windows over historical data
- Each pattern detector returns confidence scores and signal dates
- `PatternAnalyzer` aggregates signals from multiple pattern detectors
- Results include entry/exit points and confidence levels

### Performance Monitoring
- Function execution times tracked for optimization
- Data loading times monitored separately
- Memory profiling available via `psutil` and `memory-profiler`

## Common Development Tasks

### Adding a New Technical Indicator
1. Create indicator file in `indicators/` (e.g., `stochastic.py`)
2. Implement calculation function following existing patterns (RSI, MACD as reference)
3. Import in `indicators/__init__.py`
4. Add to `technical_indicators.py` if needed for batch calculation
5. Create visualization in appropriate `charts/` module

### Adding a New Chart Pattern Detector
1. Create pattern file in `indicators/patterns/` (e.g., `triangle_pattern.py`)
2. Implement detection class with `detect()` method returning signal data
3. Register in `indicators/patterns/__init__.py`
4. Add to `PatternAnalyzer` aggregation logic if needed
5. Create visualization function in `charts/pattern_charts.py`

### Training a New RL Model
1. Select investment style in dashboard (Long Term, Swing Trade, or Default)
2. Choose ticker and training date range
3. Adjust hyperparameters if needed (epochs, batch size)
4. Click "모델 학습" button
5. Model auto-saves to `models/[TICKER]_[style]_model.pth`

### Evaluating Model Performance
1. Ensure model exists in `models/` directory
2. Select test date range (should not overlap training period)
3. Click "모델 테스트" button
4. Review metrics: total return, buy & hold comparison, Sharpe ratio, trade count
5. Analyze portfolio value chart and individual trade records

## Configuration Reference

Key settings in `config.py`:
- `MODEL_DIR`: Where trained models are saved
- `INITIAL_BALANCE`: Starting capital for backtesting/RL (default: 100,000)
- `EPISODES`: Number of training episodes (default: 100)
- `BATCH_SIZE`: Training batch size (default: 64)
- `LEARNING_RATE`: Neural network learning rate (default: 0.0001)
- `RSI_WINDOW`: RSI calculation period (default: 14)
- `MACD_FAST/SLOW/SIGNAL`: MACD parameters (12/26/9)

## Codebase Characteristics

- **Language**: Python 3.8 with type hints in some modules
- **UI Framework**: Streamlit with session state management
- **RL Framework**: Custom gym environment + PyTorch DQN implementation
- **Data Source**: Yahoo Finance via `yfinance` library
- **Visualization**: Plotly for interactive charts, Matplotlib for static charts
- **Testing**: unittest framework with unit/integration separation
- **Project Structure**: Not a git repository (no version control initialized)
- **Korean Comments**: Many comments and UI labels are in Korean
