# Contributing to MikhailMarketMind

We welcome contributions to make this prediction market bot even better!

## Areas for Contribution

### 1. New Prediction Models

Add new predictors in `src/predictors/`:
- Implement the `BasePredictor` interface
- Add model-specific logic
- Update configuration to enable/disable

Example predictors to add:
- Open-source models (Llama, Mistral, etc. via Ollama)
- Ensemble methods (stacking, boosting)
- Time series models
- Statistical models

### 2. Market Analysis

Enhance analysis in `src/analyzers/`:
- Improved sentiment analysis (using ML models)
- Technical indicators
- Social signals
- News/event detection

### 3. Strategy Improvements

Improve betting strategy in `src/strategy/`:
- Alternative position sizing methods
- Multi-objective optimization
- Dynamic Kelly fraction adjustment
- Hedging strategies

### 4. Learning Enhancements

Improve learning system in `src/learning/`:
- Online learning algorithms
- Meta-learning for strategy selection
- Market-type classification
- Performance attribution

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests: `pytest tests/`
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Code Style

- Follow PEP 8
- Use type hints
- Document functions with docstrings
- Keep functions focused and modular

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Pull Request Process

1. Update README.md if needed
2. Update CHANGELOG.md with notable changes
3. Ensure CI passes
4. Get review from maintainers
5. Squash commits before merge

## Questions?

Open an issue for discussion before starting major changes.

Thank you for contributing!
