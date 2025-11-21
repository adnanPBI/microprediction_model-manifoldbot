# MikhailMarketMind: Advanced Manifold Markets Bot

An intelligent prediction market bot that participates exclusively in markets created by [MikhailTal](https://manifold.markets/MikhailTal) on Manifold Markets.

## Key Features

### 🧠 Multi-LLM Ensemble Prediction
- Combines predictions from multiple AI models (Claude, GPT-4, open-source models)
- Weighted voting based on historical accuracy
- Confidence-adjusted ensemble for robust predictions

### 📊 Advanced Market Analysis
- Liquidity and market depth analysis
- Momentum and trend detection
- Comment sentiment analysis
- Time-to-resolution considerations

### 💰 Sophisticated Betting Strategy
- Fractional Kelly betting with market impact adjustment
- Portfolio-level risk management
- Dynamic position sizing based on confidence
- Automatic rebalancing and exit strategies

### 🎯 Intelligent Learning System
- Tracks historical performance per market type
- Adjusts model weights based on accuracy
- Market-specific strategy optimization
- Continuous improvement through feedback loops

### 🏗️ Clean Architecture
- Modular, extensible design
- Comprehensive logging and monitoring
- Easy configuration management
- Robust error handling

## Architecture

```
mikhail-market-mind/
├── src/
│   ├── api/           # Manifold Markets API client
│   ├── predictors/    # Multi-LLM prediction engines
│   ├── analyzers/     # Market analysis modules
│   ├── strategy/      # Betting strategy and risk management
│   ├── learning/      # Performance tracking and optimization
│   └── bot.py         # Main bot orchestration
├── config/            # Configuration files
├── data/              # Historical data and model weights
├── tests/             # Comprehensive test suite
└── scripts/           # Utility scripts
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file with your API keys:
```bash
MANIFOLD_API_KEY=your_manifold_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
OPENAI_API_KEY=your_openai_api_key        # Optional
```

2. Configure bot parameters in `config/bot_config.yaml`

3. Set the bot's Manifold username in the config (for tracking purposes)

## Usage

### Run the bot
```bash
python -m src.bot
```

### Backtest strategies
```bash
python -m scripts.backtest
```

### View performance dashboard
```bash
python -m scripts.dashboard
```

## Bot Improvements Over manifoldbot

1. **Multi-Model Ensemble**: Instead of relying on a single LLM, we combine multiple models for more robust predictions
2. **Portfolio Optimization**: Manages risk across all positions, not just individual bets
3. **Market Intelligence**: Analyzes market microstructure, liquidity, and sentiment
4. **Adaptive Learning**: Continuously improves by learning from past performance
5. **Modular Design**: Easier to extend, test, and maintain

## Performance Tracking

The bot maintains detailed logs of:
- Individual bet performance
- Model accuracy by market type
- Portfolio returns and Sharpe ratio
- Market-specific insights

## Contributing

Contributions welcome! This project aims to push forward prediction market bot capabilities.

### Areas for Contribution
- New prediction models
- Additional market analysis techniques
- Enhanced risk management strategies
- Performance optimizations

## License

MIT License - See LICENSE file

## Acknowledgments

Built on insights from the excellent [manifoldbot](https://github.com/microprediction/manifoldbot) package.
