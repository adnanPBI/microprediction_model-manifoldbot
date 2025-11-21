# Quick Start Guide

Get MikhailMarketMind up and running in 5 minutes!

## Prerequisites

1. Python 3.8 or higher
2. Manifold Markets account
3. At least one AI API key (Claude or GPT)

## Setup Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

**Manifold Markets API Key:**
1. Go to https://manifold.markets
2. Navigate to your profile settings
3. Generate an API key
4. Note: You'll need a Manifold username for the bot

**AI Model API Keys (at least one required):**
- **Anthropic Claude**: Get from https://console.anthropic.com/
- **OpenAI GPT**: Get from https://platform.openai.com/

### 3. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your keys:
```bash
MANIFOLD_API_KEY=your_manifold_key_here
MANIFOLD_USERNAME=YourBotUsername

# At least one of these:
ANTHROPIC_API_KEY=your_claude_key_here
OPENAI_API_KEY=your_gpt_key_here
```

### 4. Adjust Configuration (Optional)

Edit `config/bot_config.yaml` to customize:
- Betting limits (min/max bet size)
- Risk parameters (max exposure, position size)
- Model weights and thresholds
- Update intervals

For first-time users, the defaults are conservative and safe.

### 5. Run the Bot

```bash
python -m src.bot
```

The bot will:
- Connect to Manifold Markets
- Fetch markets from MikhailTal
- Analyze markets using AI models
- Place bets when favorable opportunities are found
- Track performance and learn over time

## Monitoring Performance

### View Performance Dashboard

```bash
python -m scripts.dashboard
```

This shows:
- Current portfolio status
- Win rate and P&L
- Model performance
- Open positions

### Run Backtests

```bash
python -m scripts.backtest
```

Test the bot's prediction accuracy on resolved markets.

### Check Data Files

The bot stores data in the `data/` directory:
- `positions.json` - Current open positions
- `trade_history.json` - All trades executed
- `predictions.json` - Prediction history and accuracy

## Safety Features

The bot includes multiple safety features:
- **Position limits**: Max 10% of capital per market
- **Portfolio limits**: Max 50% total exposure
- **Liquidity checks**: Only trades liquid markets
- **Stop losses**: Exits losing positions
- **Confidence thresholds**: Only bets when confident

## Tuning for Performance

### Conservative (Default)
- Kelly fraction: 0.25 (quarter Kelly)
- Min confidence: 0.6
- Max exposure: 0.5

### Moderate
```yaml
betting:
  kelly_fraction: 0.5  # half Kelly

thresholds:
  min_confidence: 0.55
```

### Aggressive (Not Recommended)
```yaml
betting:
  kelly_fraction: 1.0  # full Kelly

thresholds:
  min_confidence: 0.5

risk:
  max_portfolio_exposure: 0.7
```

## Troubleshooting

### Bot Not Starting
- Check API keys are valid
- Ensure Python 3.8+ is installed
- Verify dependencies are installed

### No Bets Being Placed
- Check logs for eligibility reasons
- Markets may not meet criteria (liquidity, confidence, edge)
- Verify Manifold balance is sufficient
- Check that MikhailTal has open markets

### API Errors
- Verify API keys are correct
- Check rate limits haven't been exceeded
- Ensure network connectivity

## Next Steps

Once running:
1. Monitor performance with the dashboard
2. Review logs to understand decisions
3. Adjust configuration based on results
4. Consider running backtests to validate strategy
5. Track P&L over time

## Support

For issues:
- Check DEPLOYMENT.md for detailed setup
- Review logs in console output
- Open an issue on GitHub

Happy trading!
