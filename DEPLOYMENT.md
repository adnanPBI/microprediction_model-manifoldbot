# Deployment Guide

This guide covers deploying MikhailMarketMind in production.

## Prerequisites

- Python 3.8+
- Manifold Markets account and API key
- At least one LLM API key (Anthropic Claude or OpenAI GPT)
- Server with persistent storage (for portfolio and performance data)

## Installation

### 1. Clone and Install

```bash
git clone https://github.com/adnanPBI/microprediction_model-manifoldbot.git
cd microprediction_model-manifoldbot
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
MANIFOLD_API_KEY=your_key_here
MANIFOLD_USERNAME=YourBotUsername
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 3. Adjust Bot Configuration

Edit `config/bot_config.yaml` to tune:
- Risk parameters
- Model weights
- Betting thresholds
- Update intervals

## Running the Bot

### Local Development

```bash
python -m src.bot
```

### Production with systemd

Create `/etc/systemd/system/mikhail-bot.service`:

```ini
[Unit]
Description=MikhailMarketMind Bot
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/microprediction_model-manifoldbot
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python -m src.bot
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable mikhail-bot
sudo systemctl start mikhail-bot
sudo systemctl status mikhail-bot
```

### Production with Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.bot"]
```

Build and run:
```bash
docker build -t mikhail-bot .
docker run -d \
  --name mikhail-bot \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.env:/app/.env \
  mikhail-bot
```

## Monitoring

### View Logs

systemd:
```bash
sudo journalctl -u mikhail-bot -f
```

Docker:
```bash
docker logs -f mikhail-bot
```

### Performance Dashboard

```bash
python -m scripts.dashboard
```

### Check Portfolio

Access `data/positions.json` and `data/trade_history.json`

## Backup

Important files to backup:
- `data/positions.json` - Current positions
- `data/trade_history.json` - Trade history
- `data/predictions.json` - Prediction history
- `.env` - API keys

Setup automated backups:
```bash
#!/bin/bash
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)

tar -czf "$BACKUP_DIR/mikhail-bot-$DATE.tar.gz" \
  data/*.json \
  .env \
  config/bot_config.yaml
```

## Scaling

### Multiple Bots

Run multiple instances with different configurations:
- Different target users
- Different strategies
- Different model combinations

### API Rate Limits

Be mindful of:
- Manifold API rate limits
- LLM API rate limits
- Adjust `update_interval` in config

## Security

1. **Protect API Keys**: Never commit `.env` to git
2. **Limit Exposure**: Set reasonable `max_portfolio_exposure`
3. **Monitor Activity**: Review logs and performance regularly
4. **Backup Data**: Regular backups of critical data
5. **Update Dependencies**: Keep packages up to date

## Troubleshooting

### Bot Not Placing Bets

Check:
- API keys are valid
- Sufficient balance on Manifold
- Markets meet eligibility criteria
- Risk limits not exceeded

### High API Costs

Reduce:
- Update interval (less frequent checks)
- Number of models in ensemble
- Disable expensive analysis (sentiment, momentum)

### Poor Performance

Improve:
- Tune risk parameters
- Adjust confidence thresholds
- Review prediction accuracy
- Update model weights based on performance

## Support

For issues or questions:
- Open an issue on GitHub
- Review logs for error messages
- Check configuration settings
