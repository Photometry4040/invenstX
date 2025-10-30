# 🚀 InvenstX v1.0.0 - Initial Release

**AI-Powered Stock Trading System with Reinforcement Learning**

We're excited to announce the first major release of InvenstX, a comprehensive AI trading system featuring advanced reinforcement learning capabilities!

---

## 🌟 Highlights

### Core Features
- 🤖 **DQN-based RL Trading Agent** with 23D state space (3 basic + 18 technical indicators)
- 📊 **20+ Technical Indicators** (RSI, MACD, Bollinger Bands, ATR, etc.)
- 🔍 **Pattern Recognition Engine** (Head & Shoulders, Double Top/Bottom, Candlestick patterns)
- 📈 **Comprehensive Backtesting** with multiple strategy support
- ⚡ **Hyperparameter Optimization** powered by Optuna
- 🎨 **Interactive Streamlit Dashboard** with 5 operational modes
- 🍎 **Apple Silicon GPU Support** (MPS acceleration)

### Performance Achievements

| Model | Return | Sharpe Ratio | Trades | Status |
|-------|--------|--------------|--------|--------|
| **23D Enhanced Model** | **18.25%** | **0.4784** | 468 | ⭐ Best |
| 23D Optimized Model | 6.61% | 0.2636 | 343 | ✅ Good |
| 3D Baseline Model | 0.00% | 0.0000 | 0 | ❌ Inactive |
| Buy & Hold (Benchmark) | 19.13% | - | 1 | 📊 Reference |

**Key Result**: Our 23D Enhanced Model achieved near-parity with Buy & Hold (-0.88%p) while providing active risk management!

---

## 📦 Installation

### Quick Start

```bash
# Clone repository
git clone https://github.com/Photometry4040/invenstX.git
cd invenstX

# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run main.py
```

### Requirements
- Python 3.8+
- macOS / Linux / Windows
- 2GB+ RAM

---

## 🚀 Quick Examples

### 1. Train RL Model
```bash
python train_enhanced_model.py \
  --ticker AAPL \
  --epochs 50 \
  --batch-size 32 \
  --style default
```

### 2. Optimize Hyperparameters
```bash
python optimize_hyperparams.py \
  --ticker AAPL \
  --n-trials 30 \
  --quick
```

### 3. Evaluate Model
```bash
python evaluate_enhanced_model.py \
  --ticker AAPL \
  --style default
```

---

## 🏗️ Architecture

### DQN Network
- **Input**: 21D state vector (3 basic + 18 indicators)
- **Hidden Layers**: 256 → 256 → 64 units
- **Activation**: ReLU + Batch Normalization
- **Regularization**: Dropout (0.2-0.4)
- **Output**: 3 actions (Hold, Buy, Sell)

### Technical Indicators (18)
1. RSI, MACD, Bollinger Bands
2. Moving Averages (5, 10, 20, 50, 200)
3. ATR, Volume indicators
4. Daily Returns, Momentum, Volatility
5. Additional derived features

---

## 📚 Documentation

- **README.md**: Complete project overview with Mermaid diagrams
- **CLAUDE.md**: Development guidelines for AI assistants
- **FEATURE_ENGINEERING_GUIDE.md**: Adding new technical indicators
- **CONTRIBUTING.md**: Contribution guidelines
- **RELEASE_NOTES.md**: Detailed release information

---

## 🧪 Testing & CI/CD

- ✅ **GitHub Actions CI/CD** pipeline
- ✅ Multi-platform testing (Ubuntu, macOS)
- ✅ Python 3.8, 3.9, 3.10 support
- ✅ Automated linting (flake8, black, mypy)
- ✅ Code coverage reporting

---

## 🐛 Known Issues

1. **Optimization Time**: Hyperparameter optimization takes 30-60 minutes for 30 trials
2. **Data Limitations**: Limited to ~500 records per stock (Yahoo Finance)
3. **Overfitting Risk**: Large networks may overfit on small datasets
4. **GPU Support**: MPS (Apple Silicon) is primary; CUDA needs testing

---

## 🔮 Roadmap

### Short-term (1-2 months)
- [ ] Expand dataset to 1000+ records
- [ ] Implement Early Stopping
- [ ] Add ensemble models

### Mid-term (3-6 months)
- [ ] A3C, PPO, SAC algorithms
- [ ] Real-time trading API
- [ ] Multi-agent system

### Long-term (6-12 months)
- [ ] Transformer models
- [ ] Sentiment analysis
- [ ] Mobile app

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Past performance does not guarantee future results
- Always perform thorough backtesting
- Investment decisions are your responsibility
- Consult financial professionals before investing

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Special thanks to:
- PyTorch, Streamlit, Optuna teams
- Yahoo Finance for data access
- All contributors and supporters

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Photometry4040/invenstX/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Photometry4040/invenstX/discussions)

---

## 🎊 Thank You!

Thank you for your interest in InvenstX! We hope this tool helps you learn about algorithmic trading and reinforcement learning.

**Happy Trading! 📈**

---

**Release Date**: October 30, 2025
**Version**: 1.0.0
**Commit**: e24c116

---

## 📥 Download

Choose your preferred installation method:

- **Source Code (zip)**: Click "Source code (zip)" below
- **Source Code (tar.gz)**: Click "Source code (tar.gz)" below
- **Git Clone**: `git clone https://github.com/Photometry4040/invenstX.git`

---

*Made with ❤️ by InvenstX Team*

*AI와 함께하는 더 스마트한 투자*
