# README.md template
readme_content = """# Distributed Trading System

A modern, distributed algorithmic trading system with IBKR integration.

## Features

- **Modular Architecture**: Clean separation between data, execution, and strategy layers
- **Event-Driven**: Async event bus for component communication  
- **Risk Management**: Built-in risk controls and circuit breakers
- **Strategy Framework**: Support for conventional and ML strategies
- **Real-time Monitoring**: System metrics and alerting
- **Data Caching**: Redis-based caching for performance
- **Configuration Management**: YAML/environment-based config

## Quick Start

1. **Setup**
   ```bash
   python scripts/setup.py
   ```

2. **Start IBKR TWS/Gateway** on default ports (7497 for paper, 7496 for live)

3. **Validate Configuration**
   ```bash
   python main.py validate-config
   ```

4. **Run System**
   ```bash
   python main.py run
   ```

## Configuration

### Environment Variables (.env)
- `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`: IBKR connection
- `DATA_REDIS_URL`: Redis connection string
- `RISK_MAX_PORTFOLIO_RISK`: Maximum risk per trade (default: 2%)

### Strategy Configuration (config/strategies.yaml)
Define enabled strategies and their parameters.

## Architecture

```
├── Core Interfaces    # Abstract interfaces for all components
├── Event Bus         # Async message passing between components  
├── Data Layer        # Market data providers with caching
├── Execution Engine  # Order management and execution
├── Risk Manager      # Real-time risk monitoring and controls
├── Strategy Framework # Pluggable trading strategies
└── Monitoring        # System metrics and alerting
```

## Strategy Development

Create new strategies by extending `BaseStrategy`:

```python
from strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        # Your strategy logic here
        pass
```

## Monitoring

- System metrics collected every 30 seconds
- Automatic alerts for system issues
- Risk monitoring with circuit breakers
- Performance tracking and reporting

## Safety Features

- **Paper Trading**: Default mode for testing
- **Risk Limits**: Portfolio and position size limits
- **Circuit Breakers**: Automatic trading halt on losses
- **Connection Recovery**: Automatic reconnection to IBKR
- **Data Validation**: Input validation and error handling

## Requirements

- Python 3.8+
- IBKR TWS or IB Gateway
- Redis server
- See requirements.txt for Python packages

## License

MIT License - see LICENSE file for details.
"""

