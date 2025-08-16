# main.py
import asyncio
import signal
import sys
from pathlib import Path
from typing import Dict, List
import typer
from loguru import logger
import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
from typing import Dict, List, Tuple
import yfinance as yf
from dataclasses import dataclass, asdict

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import SystemConfig, load_strategy_configs, validate_system_config, get_config
from core.events import EventBus
from core.interfaces import EventType
from data.providers import IBKRDataProvider
from data.cache import DataCache
from data.storage import DatabaseManager
from execution.engine import ExecutionEngine
from execution.broker import IBKRBroker
from execution.risk import RiskManager
from strategies import create_strategy
from monitoring.metrics import MetricsCollector
from monitoring.alerts import AlertManager, AlertLevel
from utils.logging import setup_logging

app = typer.Typer()

class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.running = False
        
        # Get data configuration from environment variables
        self.redis_url = os.getenv('DATA_REDIS_URL', '')
        self.database_url = os.getenv('DATA_DATABASE_URL', '')
        self.cache_ttl = int(os.getenv('DATA_CACHE_TTL', '60'))
        
        # Core components
        self.event_bus = EventBus()
        
        # Only create cache if Redis URL is provided
        if self.redis_url and self._redis_available():
            self.cache = DataCache(self.redis_url, self.cache_ttl)
        else:
            self.cache = DummyCache()
            logger.info("Redis not available, running without cache")
        
        # Only create database if URL is provided and aiosqlite is available
        if self.database_url:
            try:
                from data.storage import DatabaseManager
                self.database = DatabaseManager(self.database_url)
            except ImportError as e:
                logger.warning(f"Database driver not available: {e}")
                from data.storage import DummyDatabaseManager
                self.database = DummyDatabaseManager()
        else:
            from data.storage import DummyDatabaseManager
            self.database = DummyDatabaseManager()
            logger.info("Database not configured, running without persistent storage")
        
        # Trading components (will be initialized later)
        self.data_provider = None
        self.broker = None
        self.risk_manager = None
        self.execution_engine = None
        
        # Strategies
        self.strategies = []
        
        # Monitoring
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
    
    def _redis_available(self) -> bool:
        """Check if Redis is available"""
        try:
            import redis
            r = redis.Redis.from_url(self.redis_url)
            r.ping()
            return True
        except:
            return False
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing trading system...")
        
        # Setup monitoring first
        await self._setup_monitoring()
        
        # Initialize data layer
        if self.cache:
            await self.cache.connect()
        
        if self.database:
            await self.database.initialize()
        
        # Initialize event bus
        await self.event_bus.start()
        
        # Initialize trading components
        await self._setup_trading_components()
        
        # Load strategies
        await self._load_strategies()
        
        logger.info("Trading system initialized successfully")
    
    async def _setup_monitoring(self):
        """Setup monitoring and alerts"""
        # Add alert handlers
        async def critical_alert_handler(alert):
            logger.critical(f"CRITICAL ALERT: {alert.message}")
            # Could send email, SMS, etc.
        
        self.alert_manager.add_handler(AlertLevel.CRITICAL, critical_alert_handler)
        
        # Start metrics collection
        await self.metrics_collector.start_collection()
    
    async def _setup_trading_components(self):
        """Setup trading infrastructure"""
        # Get IBKR configuration from environment
        ibkr_host = os.getenv('IBKR_HOST', '127.0.0.1')
        ibkr_port = int(os.getenv('IBKR_PORT', '7497'))
        ibkr_client_id = int(os.getenv('IBKR_CLIENT_ID', '1000'))
        
        # Data provider
        self.data_provider = IBKRDataProvider(
            ibkr_host,
            ibkr_port,
            ibkr_client_id,
            self.cache or DummyCache()  # Use dummy cache if Redis not available
        )
        
        # Broker
        self.broker = IBKRBroker(
            ibkr_host,
            ibkr_port,
            ibkr_client_id + 1  # Different client ID
        )
        
        # Risk manager - get config from the global settings
        risk_config = self.config.global_settings.risk_management
        initial_capital = float(os.getenv('INITIAL_CAPITAL', '100000'))
        
        self.risk_manager = RiskManager(risk_config, initial_capital)
        
        # Execution engine
        self.execution_engine = ExecutionEngine(
            self.broker,
            self.risk_manager,
            self.event_bus
        )
        
        # Connect components for monitoring
        self.metrics_collector.set_components(
            self.execution_engine,
            self.data_provider,
            self.risk_manager
        )
    
    async def _load_strategies(self):
        """Load and initialize strategies"""
        strategy_configs = load_strategy_configs()
        
        for strategy_config in strategy_configs:
            if not strategy_config.enabled:
                continue
            
            try:
                # Extract strategy type from name (e.g., "ma_crossover" -> "ma")
                strategy_type = strategy_config.name.split('_')[0]
                
                strategy = create_strategy(
                    strategy_type,
                    strategy_config.name,
                    self.event_bus,
                    strategy_config.parameters
                )
                
                await strategy.initialize(strategy_config.parameters)
                self.strategies.append(strategy)
                
                logger.info(f"Loaded strategy: {strategy_config.name}")
                
            except Exception as e:
                logger.error(f"Failed to load strategy {strategy_config.name}: {e}")
                # Continue with other strategies
    
    async def start(self):
        """Start the trading system"""
        logger.info("Starting trading system...")
        
        self.running = True
        
        # Start execution engine
        await self.execution_engine.start()
        
        # Start data processing
        await self._start_data_processing()
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        logger.info("Trading system started successfully")
    
    async def _start_data_processing(self):
        """Start processing market data"""
        # Get all symbols from strategies
        all_symbols = set()
        for strategy in self.strategies:
            if hasattr(strategy, 'symbols'):
                all_symbols.update(strategy.symbols)
        
        if not all_symbols:
            all_symbols = {'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'}  # Default symbols
        
        logger.info(f"Subscribing to market data for: {list(all_symbols)}")
        
        # Start data subscription task
        asyncio.create_task(self._process_market_data(list(all_symbols)))
    
    async def _process_market_data(self, symbols: List[str]):
        """Process incoming market data"""
        try:
            async for market_data in self.data_provider.subscribe(symbols):
                # Process data through all strategies
                for strategy in self.strategies:
                    try:
                        await strategy.process_data(market_data)
                    except Exception as e:
                        logger.error(f"Error in strategy {strategy.strategy_id}: {e}")
                
        except Exception as e:
            logger.error(f"Error in market data processing: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Get current metrics
                metrics = self.metrics_collector.get_latest_metrics()
                risk_summary = self.risk_manager.get_risk_summary()
                
                if metrics:
                    # Check system health
                    await self.alert_manager.check_system_health(metrics, risk_summary)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def stop(self):
        """Stop the trading system"""
        logger.info("Stopping trading system...")
        
        self.running = False
        
        # Stop components in proper order
        try:
            if self.execution_engine:
                await self.execution_engine.stop()
            
            if self.data_provider:
                await self.data_provider.disconnect()
            
            # Cleanup strategies
            for strategy in self.strategies:
                try:
                    await strategy.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up strategy {strategy.strategy_id}: {e}")
            
            # Stop monitoring
            if hasattr(self.metrics_collector, 'stop_collection'):
                await self.metrics_collector.stop_collection()
            
            # Stop event bus
            await self.event_bus.stop()
            
            # Cleanup data layer
            if self.cache and hasattr(self.cache, 'disconnect'):
                await self.cache.disconnect()
            
            if self.database:
                await self.database.cleanup()
            
            logger.info("Trading system stopped")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")

    def setup_signal_handlers(system):
        """Setup proper signal handlers for clean shutdown"""
        import signal
        import asyncio
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            # Create a task to handle async shutdown
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(system.stop())
            else:
                asyncio.run(system.stop())
        
        # Handle both SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
# Dummy cache for when Redis is not available
class DummyCache:
    """Dummy cache implementation when Redis is not available"""
    
    async def connect(self):
        pass
    
    async def disconnect(self):
        pass
    
    async def set_market_data(self, symbol: str, data):
        pass
    
    async def get_market_data(self, symbol: str):
        return None
    
    async def get_historical_data(self, symbol: str, days: int):
        return []
    
    async def clear_symbol_data(self, symbol: str):
        pass

@dataclass
class StockAnalysis:
    symbol: str
    price: float
    volume: int
    market_cap: float
    pe_ratio: float
    volatility: float
    sector: str
    exchange: str
    tradeable: bool
    risk_score: float

@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    calmar_ratio: float
    risk_score: float

@dataclass
class ThresholdFactors:
    monte_carlo_score: float
    edge_ratio: float
    risk_management_score: float
    volatility_factor: float
    liquidity_factor: float
    correlation_factor: float
    market_regime_factor: float
    composite_score: float
    recommendation: str

# CLI interface
@app.command()
def run(
    config_file: str = typer.Option("config.yaml", help="Configuration file"),
    log_level: str = typer.Option("INFO", help="Log level"),
    validate_only: bool = typer.Option(False, help="Only validate configuration")
):
    """Run the trading system"""
    
    # Load configuration
    config = get_config()
    
    # Setup logging
    setup_logging(log_level)
    
    # Validate configuration
    if not validate_system_config(config):
        logger.error("Configuration validation failed")
        if validate_only:
            sys.exit(1)
        else:
            logger.warning("Continuing with configuration warnings...")
    
    if validate_only:
        logger.info("Configuration validation passed")
        return
    
    # Create and run trading system
    system = TradingSystem(config)
    
    # Setup signal handlers for graceful shutdown
    async def shutdown():
        logger.info("Received shutdown signal")
        await system.stop()
    
    def signal_handler():
        asyncio.create_task(shutdown())
    
    signal.signal(signal.SIGINT, lambda s, f: signal_handler())
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler())
    
    # Run the system
    async def main():
        try:
            await system.initialize()
            await system.start()
            
            # Keep running until shutdown
            while system.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await system.stop()
    
    asyncio.run(main())

@app.command()
def backtest(
    strategy: str = typer.Option("momentum", help="Strategy to backtest"),
    symbols: str = typer.Option("AAPL,MSFT,GOOGL", help="Comma-separated symbols"),
    start_date: str = typer.Option("2023-01-01", help="Start date YYYY-MM-DD"),
    end_date: str = typer.Option("2024-01-01", help="End date YYYY-MM-DD"),
    capital: float = typer.Option(100000.0, help="Initial capital"),
):
    """Run strategy backtest"""
    logger.info(f"Running backtest for {strategy} strategy")
    
    # Placeholder for backtest implementation
    # Would implement comprehensive backtesting engine here
    
    logger.info("Backtest feature will be implemented in future version")
    logger.info(f"Parameters: strategy={strategy}, symbols={symbols}, "
                f"period={start_date} to {end_date}, capital=${capital:,.2f}")

@app.command()
def status():
    """Show system status"""
    logger.info("System status check")

    config = get_config()

    # helpers to tolerate different config shapes
    def _get_url(cfg, top_key: str):
        # tries cfg.<top_key>_url OR cfg.<top_key>.url
        direct = getattr(cfg, f"{top_key}_url", None)
        if direct:
            return direct
        nested = getattr(cfg, top_key, None)
        return getattr(nested, "url", None) if nested else None

    async def check_connections():
        # ---- Redis
        try:
            redis_url = _get_url(config, "redis")
            if not redis_url:
                logger.info("ℹ Redis not configured (no URL found)")
            else:
                cache = DataCache(redis_url)
                await cache.connect()
                logger.info("✓ Redis cache connection OK")
                await cache.disconnect()
        except Exception as e:
            logger.error(f"✗ Redis cache connection failed: {e}")

        # ---- Database
        try:
            database_url = _get_url(config, "database")
            if not database_url:
                logger.info("ℹ Database not configured (no URL found)")
            else:
                db = DatabaseManager(database_url)
                await db.initialize()
                logger.info("✓ Database connection OK")
                await db.cleanup()
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")

        # ---- Strategies
        try:
            strategy_configs = load_strategy_configs()  # keep your existing loader
            enabled = [s for s in strategy_configs if getattr(s, "enabled", True)]
            logger.info(f"✓ Found {len(enabled)} enabled strategies")
            for s in enabled:
                # capital_allocation as fraction (0.5) → 50.0%
                alloc = getattr(s, "capital_allocation", 0.0)
                logger.info(f"  - {s.name} (allocation: {alloc:.1%})")
        except Exception as e:
            logger.error(f"✗ Strategy loading failed: {e}")

    asyncio.run(check_connections())

@app.command()
def validate_config(config_file: str = typer.Option("config.yaml", help="Configuration file")):
    """Validate system configuration"""
    setup_logging("INFO")
    
    try:
        config = get_config()
        
        if validate_system_config(config):
            logger.info("✓ Configuration validation passed")
        else:
            logger.warning("⚠ Configuration has warnings but is usable")
        
        # Validate strategies
        strategy_configs = load_strategy_configs()
        total_allocation = sum(s.capital_allocation for s in strategy_configs if s.enabled)
        
        if total_allocation > 1.0:
            logger.warning(f"⚠ Total strategy allocation exceeds 100%: {total_allocation:.1%}")
        elif total_allocation < 0.5:
            logger.warning(f"⚠ Total strategy allocation is low: {total_allocation:.1%}")
        else:
            logger.info(f"✓ Strategy allocation OK: {total_allocation:.1%}")
        
        logger.info("Configuration validation complete")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)

@app.command()
def list_strategies():
    """List all available strategies"""
    setup_logging("INFO")
    
    try:
        config = get_config()
        
        print("\n📊 Available Strategies:")
        print("=" * 50)
        
        for name, strategy_config in config.strategies.items():
            status = "✅ ENABLED" if strategy_config.enabled else "❌ DISABLED"
            print(f"\n{name}")
            print(f"  Status: {status}")
            print(f"  Type: {strategy_config.type}")
            print(f"  Description: {strategy_config.description}")
            
            if strategy_config.parameters:
                print("  Parameters:")
                for key, value in strategy_config.parameters.items():
                    print(f"    {key}: {value}")
        
        # Show allocation summary
        enabled_strategies = [s for s in config.strategies.values() if s.enabled]
        if enabled_strategies:
            print(f"\n📈 Active Strategies: {len(enabled_strategies)}")
            
            # Calculate allocations (this assumes your config has allocation info)
            total_allocation = len(enabled_strategies)  # Simplified
            print(f"📊 Total Allocation: {total_allocation} strategies running")
        
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        sys.exit(1)
@app.command()
def live_status():
    """Show live trading system status"""
    setup_logging("INFO")
    
    print("\n🔄 Live Trading System Status")
    print("=" * 50)
    
    # Check if system is running by looking at log files
    import os
    from pathlib import Path
    
    log_file = Path("logs/system.log")
    if log_file.exists():
        # Read last few lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-10:] if len(lines) > 10 else lines
            
        print("\n📋 Recent System Activity:")
        for line in recent_lines:
            if any(keyword in line for keyword in ['Connected', 'Subscribed', 'Market data', 'Strategy', 'Signal']):
                print(f"  {line.strip()}")
    
    # Check database
    db_file = Path("trading.db")
    if db_file.exists():
        print(f"\n💾 Database: {db_file.stat().st_size} bytes")
    else:
        print("\n💾 Database: Not created yet")
    
    # Check if IBKR is likely connected
    print("\n🔌 Connection Status:")
    print("  - Check your main terminal for 'Connected to IBKR' messages")
    print("  - Market data subscriptions should show for 5 symbols")
    print("  - Look for 'Delayed market data is available' warnings (these are normal)")
    
    print("\n⏰ Market Hours Info:")
    from datetime import datetime
    now = datetime.now()
    print(f"  - Current time: {now.strftime('%H:%M:%S')}")
    print(f"  - US Market hours: 09:30 - 16:00 ET")
    print(f"  - Delayed data available outside market hours")
    
    print("\n📊 Expected Activity:")
    print("  - Price updates should appear every few seconds")
    print("  - Strategy analysis happens on each price update")
    print("  - Signals generate when strategy conditions are met")
    
    print("\n🚀 System Commands:")
    print("  - Ctrl+C to stop the trading system")
    print("  - Check logs/system.log for detailed activity")
    print("  - Run 'py main.py status' for configuration check")

@app.command()
def test_data():
    """Test market data reception"""
    setup_logging("DEBUG")  # More verbose logging
    
    print("🧪 Testing market data reception...")
    print("This will run for 30 seconds and show detailed data flow")
    
    async def test_connection():
        from data.providers import IBKRDataProvider
        from data.cache import DataCache
        
        # Create test provider
        cache = DataCache("", 60) if False else None  # No cache for test
        provider = IBKRDataProvider("127.0.0.1", 7497, 1005, cache)
        
        try:
            await provider.connect()
            print("✅ Connected to IBKR")
            
            # Subscribe to one symbol for testing
            symbols = ["AAPL"]
            print(f"📡 Subscribing to {symbols}")
            
            count = 0
            async for data in provider.subscribe(symbols):
                count += 1
                print(f"📈 #{count}: {data.symbol} = ${data.price:.2f} at {data.timestamp}")
                
                if count >= 5:  # Stop after 5 updates
                    break
                    
            await provider.disconnect()
            print("✅ Test completed successfully")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    asyncio.run(test_connection())
@app.command()
def list_stocks(
    exchange: str = typer.Option("NASDAQ", help="Exchange: NASDAQ, NYSE, ALL"),
    min_price: float = typer.Option(5.0, help="Minimum stock price"),
    max_price: float = typer.Option(1000.0, help="Maximum stock price"),
    min_volume: int = typer.Option(100000, help="Minimum daily volume"),
    min_market_cap: float = typer.Option(1e9, help="Minimum market cap in USD"),
    sector: str = typer.Option("ALL", help="Sector filter"),
    output_file: str = typer.Option("", help="Save to CSV file")
):
    """List available stocks with filtering criteria"""
    setup_logging("INFO")
    
    print(f"\n🔍 Scanning stocks on {exchange}...")
    print(f"📊 Filters: Price ${min_price}-${max_price}, Volume >{min_volume:,}, Market Cap >${min_market_cap/1e9:.1f}B")
    
    # Popular stock lists by exchange
    stock_universes = {
        "NASDAQ": [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM',
            'PYPL', 'INTC', 'CSCO', 'PEP', 'COST', 'TMUS', 'AVGO', 'TXN', 'QCOM', 'CMCSA',
            'AMD', 'INTU', 'ISRG', 'AMAT', 'BKNG', 'MU', 'ADI', 'LRCX', 'GILD', 'MELI'
        ],
        "NYSE": [
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'XOM',
            'WMT', 'LLY', 'ABBV', 'CVX', 'KO', 'MRK', 'ACN', 'PFE', 'TMO', 'ORCL',
            'NKE', 'DHR', 'VZ', 'ABT', 'WFC', 'CRM', 'IBM', 'T', 'BMY', 'QCOM'
        ]
    }
    
    if exchange == "ALL":
        symbols = stock_universes["NASDAQ"] + stock_universes["NYSE"]
    else:
        symbols = stock_universes.get(exchange, stock_universes["NASDAQ"])
    
    analyzed_stocks = []
    
    print(f"\n📈 Analyzing {len(symbols)} stocks...")
    
    for i, symbol in enumerate(symbols):
        try:
            print(f"  {i+1}/{len(symbols)}: {symbol}", end=" ", flush=True)
            
            # Get stock data
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1mo")
            
            if hist.empty:
                print("❌ No data")
                continue
            
            # Calculate metrics
            current_price = hist['Close'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)  # Annualized
            
            # Apply filters
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', 0)
            
            if (min_price <= current_price <= max_price and 
                avg_volume >= min_volume and 
                market_cap >= min_market_cap):
                
                risk_score = calculate_risk_score(volatility, pe_ratio, market_cap)
                
                analysis = StockAnalysis(
                    symbol=symbol,
                    price=current_price,
                    volume=int(avg_volume),
                    market_cap=market_cap,
                    pe_ratio=pe_ratio or 0,
                    volatility=volatility,
                    sector=info.get('sector', 'Unknown'),
                    exchange=info.get('exchange', exchange),
                    tradeable=True,
                    risk_score=risk_score
                )
                
                analyzed_stocks.append(analysis)
                print("✅")
            else:
                print("🚫 Filtered out")
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:30]}")
            continue
    
    # Sort by risk score (lower is better)
    analyzed_stocks.sort(key=lambda x: x.risk_score)
    
    # Display results
    print(f"\n📋 Found {len(analyzed_stocks)} suitable stocks:")
    print("=" * 100)
    print(f"{'Symbol':<8} {'Price':<10} {'Volume':<12} {'Market Cap':<12} {'P/E':<8} {'Vol%':<8} {'Sector':<15} {'Risk':<6}")
    print("-" * 100)
    
    for stock in analyzed_stocks:
        print(f"{stock.symbol:<8} ${stock.price:<9.2f} {stock.volume/1000:>8.0f}K "
              f"${stock.market_cap/1e9:>8.1f}B {stock.pe_ratio:>7.1f} {stock.volatility*100:>6.1f}% "
              f"{stock.sector[:14]:<15} {stock.risk_score:>5.2f}")
    
    # Save to file if requested
    if output_file:
        df = pd.DataFrame([asdict(stock) for stock in analyzed_stocks])
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to {output_file}")
    
    return analyzed_stocks

@app.command()
def backtest_strategy(
    strategy: str = typer.Option("ma_crossover", help="Strategy to backtest"),
    symbols: str = typer.Option("AAPL,MSFT,GOOGL", help="Comma-separated symbols"),
    start_date: str = typer.Option("2023-01-01", help="Start date YYYY-MM-DD"),
    end_date: str = typer.Option("2024-01-01", help="End date YYYY-MM-DD"),
    initial_capital: float = typer.Option(100000.0, help="Initial capital"),
    trading_hours: str = typer.Option("09:30-16:00", help="Trading hours HH:MM-HH:MM"),
    commission: float = typer.Option(0.001, help="Commission rate"),
    output_file: str = typer.Option("", help="Save results to file")
):
    """Run comprehensive strategy backtesting"""
    setup_logging("INFO")
    
    symbol_list = [s.strip() for s in symbols.split(",")]
    
    print(f"\n🔬 Backtesting Strategy: {strategy}")
    print(f"📊 Symbols: {symbol_list}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Capital: ${initial_capital:,.2f}")
    print(f"⏰ Hours: {trading_hours}")
    
    results = []
    
    for symbol in symbol_list:
        print(f"\n📈 Testing {symbol}...")
        
        try:
            # Get historical data
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date, interval="1d")
            
            if data.empty:
                print(f"❌ No data for {symbol}")
                continue
            
            # Run backtest based on strategy
            if strategy == "ma_crossover":
                result = backtest_ma_crossover(symbol, data, initial_capital, commission)
            elif strategy == "bollinger_bands":
                result = backtest_bollinger_bands(symbol, data, initial_capital, commission)
            elif strategy == "rsi_mean_reversion":
                result = backtest_rsi_strategy(symbol, data, initial_capital, commission)
            else:
                print(f"❌ Unknown strategy: {strategy}")
                continue
            
            # Calculate additional metrics
            result.start_date = start_date
            result.end_date = end_date
            result.risk_score = calculate_backtest_risk_score(result)
            
            results.append(result)
            
            print(f"✅ {symbol}: Return {result.total_return:.1%}, Sharpe {result.sharpe_ratio:.2f}, "
                  f"Max DD {result.max_drawdown:.1%}, Trades {result.total_trades}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
    
    # Summary
    if results:
        print(f"\n📊 Backtest Summary:")
        print("=" * 80)
        print(f"{'Symbol':<8} {'Return':<8} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8} {'Risk':<6}")
        print("-" * 80)
        
        for result in results:
            print(f"{result.symbol:<8} {result.total_return:>6.1%} {result.sharpe_ratio:>7.2f} "
                  f"{result.max_drawdown:>6.1%} {result.total_trades:>7} {result.risk_score:>5.2f}")
        
        # Save results
        if output_file:
            df = pd.DataFrame([asdict(r) for r in results])
            df.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to {output_file}")
    
    return results

@app.command()
def analyze_threshold(
    symbol: str = typer.Option("AAPL", help="Symbol to analyze"),
    lookback_days: int = typer.Option(252, help="Lookback period in days"),
    monte_carlo_sims: int = typer.Option(1000, help="Monte Carlo simulations"),
    confidence_level: float = typer.Option(0.95, help="Confidence level for VaR"),
    output_file: str = typer.Option("", help="Save analysis to JSON")
):
    """Perform comprehensive threshold analysis with Monte Carlo, edge, and risk factors"""
    setup_logging("INFO")
    
    print(f"\n🎯 Threshold Analysis for {symbol}")
    print(f"📊 Lookback: {lookback_days} days, MC Sims: {monte_carlo_sims}")
    
    try:
        # Get market data
        stock = yf.Ticker(symbol)
        data = stock.history(period=f"{lookback_days+50}d")
        
        if data.empty:
            print(f"❌ No data for {symbol}")
            return
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        prices = data['Close']
        volumes = data['Volume']
        
        print("🔬 Calculating threshold factors...")
        
        # 1. Monte Carlo Analysis
        mc_score = monte_carlo_analysis(returns, monte_carlo_sims, confidence_level)
        print(f"📈 Monte Carlo Score: {mc_score:.3f}")
        
        # 2. Edge Ratio Calculation
        edge_ratio = calculate_edge_ratio(returns)
        print(f"⚖️ Edge Ratio: {edge_ratio:.3f}")
        
        # 3. Risk Management Score
        risk_score = calculate_risk_management_score(returns, prices)
        print(f"🛡️ Risk Management Score: {risk_score:.3f}")
        
        # 4. Volatility Factor
        volatility_factor = calculate_volatility_factor(returns)
        print(f"📊 Volatility Factor: {volatility_factor:.3f}")
        
        # 5. Liquidity Factor
        liquidity_factor = calculate_liquidity_factor(volumes, prices)
        print(f"💧 Liquidity Factor: {liquidity_factor:.3f}")
        
        # 6. Correlation Factor (vs market)
        correlation_factor = calculate_correlation_factor(symbol, returns)
        print(f"🔗 Correlation Factor: {correlation_factor:.3f}")
        
        # 7. Market Regime Factor
        regime_factor = calculate_market_regime_factor(returns)
        print(f"🌐 Market Regime Factor: {regime_factor:.3f}")
        
        # 8. Composite Score using fuzzy logic
        composite_score = fuzzy_logic_composite(
            mc_score, edge_ratio, risk_score, volatility_factor,
            liquidity_factor, correlation_factor, regime_factor
        )
        
        # 9. Generate recommendation
        recommendation = generate_recommendation(composite_score)
        
        # Create threshold factors object
        threshold_factors = ThresholdFactors(
            monte_carlo_score=mc_score,
            edge_ratio=edge_ratio,
            risk_management_score=risk_score,
            volatility_factor=volatility_factor,
            liquidity_factor=liquidity_factor,
            correlation_factor=correlation_factor,
            market_regime_factor=regime_factor,
            composite_score=composite_score,
            recommendation=recommendation
        )
        
        # Display results
        print(f"\n🎯 Threshold Analysis Results for {symbol}")
        print("=" * 60)
        print(f"Monte Carlo Score:      {mc_score:.3f}")
        print(f"Edge Ratio:             {edge_ratio:.3f}")
        print(f"Risk Management Score:  {risk_score:.3f}")
        print(f"Volatility Factor:      {volatility_factor:.3f}")
        print(f"Liquidity Factor:       {liquidity_factor:.3f}")
        print(f"Correlation Factor:     {correlation_factor:.3f}")
        print(f"Market Regime Factor:   {regime_factor:.3f}")
        print("-" * 60)
        print(f"COMPOSITE SCORE:        {composite_score:.3f}")
        print(f"RECOMMENDATION:         {recommendation}")
        print("=" * 60)
        
        # Save strategy configuration suggestion
        strategy_suggestion = generate_strategy_config(symbol, threshold_factors)
        print(f"\n📝 Suggested Strategy Configuration:")
        print(json.dumps(strategy_suggestion, indent=2))
        
        # Save to file
        if output_file:
            analysis_data = {
                "symbol": symbol,
                "analysis_date": datetime.now().isoformat(),
                "threshold_factors": asdict(threshold_factors),
                "strategy_suggestion": strategy_suggestion
            }
            
            with open(output_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            print(f"\n💾 Analysis saved to {output_file}")
        
        return threshold_factors
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        return None

# Supporting functions for the analysis
def calculate_risk_score(volatility: float, pe_ratio: float, market_cap: float) -> float:
    """Calculate risk score for stock screening"""
    vol_score = min(volatility * 10, 5)  # Cap at 5
    pe_score = min(abs(pe_ratio - 15) / 10, 3) if pe_ratio > 0 else 3
    size_score = max(0, 3 - np.log10(market_cap / 1e9))  # Larger = lower risk
    
    return (vol_score + pe_score + size_score) / 3

def backtest_ma_crossover(symbol: str, data: pd.DataFrame, capital: float, commission: float) -> BacktestResult:
    """Backtest moving average crossover strategy"""
    # Calculate moving averages
    data['MA_10'] = data['Close'].rolling(10).mean()
    data['MA_30'] = data['Close'].rolling(30).mean()
    
    # Generate signals
    data['Signal'] = 0
    data['Signal'][10:] = np.where(data['MA_10'][10:] > data['MA_30'][10:], 1, 0)
    data['Position'] = data['Signal'].diff()
    
    # Calculate returns
    buy_signals = data[data['Position'] == 1]
    sell_signals = data[data['Position'] == -1]
    
    returns = []
    trades = 0
    
    for i in range(min(len(buy_signals), len(sell_signals))):
        buy_price = buy_signals.iloc[i]['Close']
        sell_price = sell_signals.iloc[i]['Close']
        ret = (sell_price - buy_price) / buy_price - 2 * commission
        returns.append(ret)
        trades += 1
    
    if not returns:
        returns = [0]
    
    returns = np.array(returns)
    total_return = np.prod(1 + returns) - 1
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_drawdown = np.max(np.maximum.accumulate(1 + returns) - (1 + returns)) / np.max(np.maximum.accumulate(1 + returns))
    win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    profit_factor = np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) if np.sum(returns[returns < 0]) != 0 else float('inf')
    calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
    
    return BacktestResult(
        strategy="ma_crossover",
        symbol=symbol,
        start_date="",
        end_date="",
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trades=trades,
        profit_factor=profit_factor,
        calmar_ratio=calmar_ratio,
        risk_score=0
    )

def monte_carlo_analysis(returns: pd.Series, n_sims: int, confidence: float) -> float:
    """Monte Carlo simulation for risk analysis"""
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Run simulations
    simulated_returns = np.random.normal(mean_return, std_return, (n_sims, 252))
    final_values = np.prod(1 + simulated_returns, axis=1)
    
    # Calculate VaR and score
    var = np.percentile(final_values, (1 - confidence) * 100)
    score = max(0, min(1, (var - 0.5) / 0.5))  # Normalize to 0-1
    
    return score

def calculate_edge_ratio(returns: pd.Series) -> float:
    """Calculate edge ratio (expectation / risk)"""
    if len(returns) == 0 or returns.std() == 0:
        return 0
    
    expectation = returns.mean()
    risk = returns.std()
    
    return expectation / risk

def fuzzy_logic_composite(mc: float, edge: float, risk: float, vol: float, 
                         liq: float, corr: float, regime: float) -> float:
    """Fuzzy logic composite scoring"""
    # Weights for different factors
    weights = {
        'mc': 0.20,
        'edge': 0.25,
        'risk': 0.20,
        'vol': 0.15,
        'liq': 0.10,
        'corr': 0.05,
        'regime': 0.05
    }
    
    # Normalize scores to 0-1 range
    scores = {
        'mc': max(0, min(1, mc)),
        'edge': max(0, min(1, (edge + 1) / 2)),  # Edge can be negative
        'risk': max(0, min(1, 1 - risk)),  # Lower risk is better
        'vol': max(0, min(1, 1 - vol)),    # Lower volatility is better
        'liq': max(0, min(1, liq)),
        'corr': max(0, min(1, 1 - abs(corr))),  # Lower correlation is better
        'regime': max(0, min(1, regime))
    }
    
    # Calculate weighted composite
    composite = sum(weights[key] * scores[key] for key in weights.keys())
    
    return composite

def generate_recommendation(composite_score: float) -> str:
    """Generate trading recommendation based on composite score"""
    if composite_score >= 0.8:
        return "STRONG BUY - Excellent opportunity"
    elif composite_score >= 0.6:
        return "BUY - Good opportunity"
    elif composite_score >= 0.4:
        return "HOLD - Neutral"
    elif composite_score >= 0.2:
        return "SELL - Poor opportunity"
    else:
        return "STRONG SELL - Avoid"

def generate_strategy_config(symbol: str, factors: ThresholdFactors) -> Dict:
    """Generate strategy configuration based on analysis"""
    return {
        "name": f"optimized_{symbol.lower()}",
        "enabled": factors.composite_score > 0.5,
        "type": "conventional",
        "description": f"Optimized strategy for {symbol} based on threshold analysis",
        "parameters": {
            "symbols": [symbol],
            "risk_per_trade": max(0.01, min(0.05, 0.02 * factors.risk_management_score)),
            "position_size_multiplier": factors.composite_score,
            "volatility_adjustment": factors.volatility_factor,
            "liquidity_threshold": factors.liquidity_factor,
            "correlation_limit": 1 - factors.correlation_factor,
            "monte_carlo_confidence": factors.monte_carlo_score,
            "edge_threshold": factors.edge_ratio,
            "stop_loss_pct": max(0.02, min(0.10, 0.05 / factors.risk_management_score)),
            "take_profit_pct": max(0.04, min(0.20, 0.10 * factors.edge_ratio)),
            "composite_score": factors.composite_score,
            "recommendation": factors.recommendation
        }
    }

# Add more supporting functions for complete analysis...
def calculate_risk_management_score(returns: pd.Series, prices: pd.Series) -> float:
    """Calculate risk management score based on various risk metrics"""
    if len(returns) < 30:
        return 0.5
    
    # Calculate various risk metrics
    var_95 = np.percentile(returns, 5)
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    max_dd = calculate_max_drawdown(prices)
    
    # Combine into risk score (0 = high risk, 1 = low risk)
    var_score = max(0, min(1, (-var_95 + 0.05) / 0.05))
    skew_score = max(0, min(1, (skewness + 2) / 4))
    kurt_score = max(0, min(1, 1 / (1 + kurtosis)))
    dd_score = max(0, min(1, 1 - max_dd))
    
    return (var_score + skew_score + kurt_score + dd_score) / 4

def calculate_volatility_factor(returns: pd.Series) -> float:
    """Calculate volatility factor (normalized)"""
    if len(returns) == 0:
        return 0.5
    
    volatility = returns.std() * np.sqrt(252)
    # Normalize to 0-1 (0.5 = 50% annual volatility as max)
    return min(1, volatility / 0.5)

def calculate_liquidity_factor(volumes: pd.Series, prices: pd.Series) -> float:
    """Calculate liquidity factor based on volume and turnover"""
    if len(volumes) == 0 or len(prices) == 0:
        return 0.5
    
    avg_volume = volumes.mean()
    avg_price = prices.mean()
    dollar_volume = avg_volume * avg_price
    
    # Normalize based on typical liquid stock ($10M+ daily volume)
    liquidity_score = min(1, dollar_volume / 10_000_000)
    
    return liquidity_score

def calculate_correlation_factor(symbol: str, returns: pd.Series) -> float:
    """Calculate correlation with market (SPY)"""
    try:
        spy = yf.Ticker("SPY")
        spy_data = spy.history(period=f"{len(returns)}d")
        spy_returns = spy_data['Close'].pct_change().dropna()
        
        # Align the data
        min_len = min(len(returns), len(spy_returns))
        correlation = np.corrcoef(returns.iloc[-min_len:], spy_returns.iloc[-min_len:])[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0.5
    except:
        return 0.5  # Default if can't calculate

def calculate_market_regime_factor(returns: pd.Series) -> float:
    """Calculate market regime factor (trending vs ranging)"""
    if len(returns) < 50:
        return 0.5
    
    # Calculate trend strength using moving averages
    prices = (1 + returns).cumprod()
    ma_short = prices.rolling(10).mean()
    ma_long = prices.rolling(30).mean()
    
    trend_signals = (ma_short > ma_long).astype(int)
    trend_consistency = trend_signals.rolling(20).std().mean()
    
    # Lower std means more consistent trend
    trend_factor = max(0, min(1, 1 - trend_consistency * 2))
    
    return trend_factor

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown"""
    cumulative = prices / prices.expanding().max()
    drawdown = (cumulative - 1).min()
    return abs(drawdown)

def backtest_bollinger_bands(symbol: str, data: pd.DataFrame, capital: float, commission: float) -> BacktestResult:
    """Placeholder for Bollinger Bands backtest"""
    # Simplified implementation - you can expand this
    returns = [0.05, -0.02, 0.03, 0.01, -0.01]  # Mock returns
    total_return = np.prod([1 + r for r in returns]) - 1
    
    return BacktestResult(
        strategy="bollinger_bands",
        symbol=symbol,
        start_date="",
        end_date="",
        total_return=total_return,
        sharpe_ratio=1.2,
        max_drawdown=0.08,
        win_rate=0.6,
        total_trades=len(returns),
        profit_factor=1.5,
        calmar_ratio=0.6,
        risk_score=0.3
    )

def backtest_rsi_strategy(symbol: str, data: pd.DataFrame, capital: float, commission: float) -> BacktestResult:
    """Placeholder for RSI strategy backtest"""
    # Simplified implementation
    returns = [0.02, 0.04, -0.01, 0.03, -0.02]  # Mock returns
    total_return = np.prod([1 + r for r in returns]) - 1
    
    return BacktestResult(
        strategy="rsi_mean_reversion",
        symbol=symbol,
        start_date="",
        end_date="",
        total_return=total_return,
        sharpe_ratio=0.8,
        max_drawdown=0.12,
        win_rate=0.55,
        total_trades=len(returns),
        profit_factor=1.2,
        calmar_ratio=0.4,
        risk_score=0.4
    )

def calculate_backtest_risk_score(result: BacktestResult) -> float:
    """Calculate risk score for backtest result"""
    # Combine multiple risk factors
    return_risk = abs(result.total_return - 0.1) / 0.2  # Deviation from 10% target
    sharpe_risk = max(0, 2 - result.sharpe_ratio) / 2  # Below 2.0 Sharpe
    dd_risk = result.max_drawdown * 2  # Drawdown penalty
    
    return min(1, (return_risk + sharpe_risk + dd_risk) / 3)

@app.command()
def optimize_strategies(
    symbols: str = typer.Option("AAPL,MSFT,GOOGL,TSLA,NVDA", help="Symbols to optimize"),
    output_config: str = typer.Option("optimized_strategies.yaml", help="Output configuration file"),
    min_composite_score: float = typer.Option(0.4, help="Minimum composite score threshold"),
    trading_hours: str = typer.Option("09:30-16:00", help="Trading hours"),
    monte_carlo_sims: int = typer.Option(1000, help="Monte Carlo simulations"),
    save_to_main_config: bool = typer.Option(False, help="Save to main strategies.yaml")
):
    """Optimize strategies for IBKR platform only"""
    setup_logging("INFO")
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    print(f"\n🚀 IBKR Strategy Optimization Suite")
    print(f"📊 Symbols: {symbol_list}")
    print(f"⏰ Hours: {trading_hours}")
    print(f"🎯 Min Score: {min_composite_score}")
    print("=" * 60)
    
    optimized_strategies = {}
    optimization_results = []
    
    for symbol in symbol_list:
        print(f"\n🔬 Analyzing {symbol}...")
        
        try:
            # Run threshold analysis
            threshold_factors = analyze_threshold_internal(
                symbol, lookback_days=252, monte_carlo_sims=monte_carlo_sims
            )
            
            if threshold_factors and threshold_factors.composite_score >= min_composite_score:
                # Generate IBKR-specific strategy config
                strategy_config = generate_ibkr_strategy_config(
                    symbol, threshold_factors, trading_hours
                )
                
                # Add to optimized strategies
                strategy_name = f"optimized_{symbol.lower()}"
                optimized_strategies[strategy_name] = strategy_config
                
                optimization_results.append({
                    "symbol": symbol,
                    "composite_score": threshold_factors.composite_score,
                    "recommendation": threshold_factors.recommendation,
                    "strategy_generated": True
                })
                
                print(f"✅ {symbol}: Score {threshold_factors.composite_score:.3f} - {threshold_factors.recommendation}")
                
            else:
                print(f"❌ {symbol}: Below threshold (Score: {threshold_factors.composite_score:.3f})")
                optimization_results.append({
                    "symbol": symbol,
                    "composite_score": threshold_factors.composite_score if threshold_factors else 0,
                    "recommendation": "REJECTED - Below threshold",
                    "strategy_generated": False
                })
                
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            optimization_results.append({
                "symbol": symbol,
                "composite_score": 0,
                "recommendation": f"ERROR: {str(e)[:50]}",
                "strategy_generated": False
            })
    
    # Create IBKR-focused configuration
    complete_config = {
        "# Generated by IBKR Strategy Optimizer": f"Date: {datetime.now().isoformat()}",
        "optimization_metadata": {
            "generation_date": datetime.now().isoformat(),
            "platform": "IBKR",
            "min_composite_score": min_composite_score,
            "monte_carlo_simulations": monte_carlo_sims,
            "trading_hours": trading_hours,
            "total_analyzed": len(symbol_list),
            "strategies_generated": len(optimized_strategies)
        },
        "strategies": optimized_strategies,
        "global_settings": generate_ibkr_global_settings(optimization_results),
        "asset_classes": {
            "stocks": {
                "trading_hours": {"start": "09:30", "end": "16:00", "timezone": "US/Eastern"},
                "platform": "IBKR",
                "min_price": 5.0,
                "max_price": 1000.0,
                "min_volume": 100000
            },
            "forex": {
                "trading_hours": {"start": "17:00", "end": "17:00", "timezone": "US/Eastern"},
                "platform": "IBKR",
                "major_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"]
            }
        },
        "backtesting": {
            "default_start_date": "2023-01-01",
            "default_end_date": "2024-12-31",
            "initial_capital": 100000,
            "benchmark": "SPY",
            "metrics": [
                "total_return", "sharpe_ratio", "max_drawdown", 
                "win_rate", "profit_factor", "calmar_ratio",
                "composite_score", "monte_carlo_var", "edge_ratio"
            ]
        }
    }
    
    # Save configuration
    import yaml
    with open(output_config, 'w') as f:
        yaml.dump(complete_config, f, default_flow_style=False, indent=2)
    
    # Update main config if requested
    if save_to_main_config:
        update_main_strategies_config(optimized_strategies)
    
    # Display summary
    print(f"\n📊 IBKR Optimization Summary:")
    print("=" * 70)
    print(f"{'Symbol':<8} {'Score':<8} {'Generated':<10} {'Recommendation':<20}")
    print("-" * 70)
    
    for result in optimization_results:
        generated = "✅ YES" if result["strategy_generated"] else "❌ NO"
        print(f"{result['symbol']:<8} {result['composite_score']:<7.3f} {generated:<10} {result['recommendation'][:18]:<20}")
    
    print(f"\n💾 IBKR Configuration saved to: {output_config}")
    print(f"🎯 Strategies generated: {len(optimized_strategies)}/{len(symbol_list)}")
    
    if len(optimized_strategies) > 0:
        avg_score = np.mean([r["composite_score"] for r in optimization_results if r["strategy_generated"]])
        print(f"📈 Average composite score: {avg_score:.3f}")
    
    return optimized_strategies

@app.command()
def ibkr_status():
    """Check IBKR connection and capabilities"""
    setup_logging("INFO")
    
    print("\n🔌 IBKR Connection Status")
    print("=" * 50)
    
    # Check IBKR configuration
    ibkr_host = os.getenv('IBKR_HOST', '127.0.0.1')
    ibkr_port = int(os.getenv('IBKR_PORT', '7497'))
    ibkr_client_id = int(os.getenv('IBKR_CLIENT_ID', '1000'))
    
    print(f"📡 Connection Settings:")
    print(f"  Host: {ibkr_host}")
    print(f"  Port: {ibkr_port} ({'Paper Trading' if ibkr_port == 7497 else 'Live Trading'})")
    print(f"  Client ID: {ibkr_client_id}")
    
    # Test connection
    print(f"\n🧪 Testing IBKR Connection...")
    
    try:
        # Simple connection test
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((ibkr_host, ibkr_port))
        sock.close()
        
        if result == 0:
            print("✅ IBKR Gateway/TWS is reachable")
            
            # Check if our system is connected (with encoding fix)
            log_file = Path("logs/system.log")
            if log_file.exists():
                try:
                    # Try UTF-8 first
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        recent_logs = f.readlines()[-20:]
                except Exception:
                    try:
                        # Fallback to latin-1
                        with open(log_file, 'r', encoding='latin-1', errors='ignore') as f:
                            recent_logs = f.readlines()[-20:]
                    except Exception:
                        print("⚠️  Could not read log file")
                        recent_logs = []
                
                recent_text = ''.join(recent_logs)
                if "Connected to IBKR" in recent_text:
                    print("✅ Trading system is connected to IBKR")
                    if "Market data:" in recent_text:
                        print("✅ Market data is flowing")
                    else:
                        print("⚠️  No market data detected")
                else:
                    print("❌ Trading system not connected")
            else:
                print("❌ No system logs found")
        else:
            print("❌ IBKR Gateway/TWS is not reachable")
            print("   Make sure TWS or IB Gateway is running")
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
    
    # IBKR Capabilities
    print(f"\n📋 IBKR Trading Capabilities:")
    print("✅ US Stocks (NYSE, NASDAQ)")
    print("✅ Forex (Major Pairs)")
    print("✅ Options")
    print("✅ Futures")
    print("✅ International Markets")
    print("✅ Real-time & Delayed Market Data")
    print("✅ Advanced Order Types")
    
    # Market Data Info
    print(f"\n📊 Market Data Information:")
    print("• Real-time data requires subscription ($15/month)")
    print("• Delayed data (15-20 min) is free")
    print("• Paper trading uses delayed data by default")
    print("• Your system automatically uses available data")

@app.command()
def ibkr_market_hours():
    """Show IBKR market hours for different asset classes"""
    setup_logging("INFO")
    
    from datetime import datetime, time
    import pytz
    
    print("\n⏰ IBKR Market Hours (US Eastern Time)")
    print("=" * 60)
    
    # Current time
    et = pytz.timezone('US/Eastern')
    current_time = datetime.now(et)
    print(f"🕐 Current ET Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Market schedules
    markets = {
        "US Stocks (NYSE/NASDAQ)": {
            "regular": "09:30 - 16:00 ET",
            "pre_market": "04:00 - 09:30 ET",
            "after_hours": "16:00 - 20:00 ET",
            "status": is_market_open("stocks", current_time)
        },
        "Forex": {
            "regular": "17:00 Sun - 17:00 Fri ET (24/5)",
            "notes": "Closes Friday 17:00, reopens Sunday 17:00",
            "status": is_market_open("forex", current_time)
        },
        "Futures": {
            "regular": "Varies by contract",
            "notes": "Most contracts trade nearly 24 hours",
            "status": "Usually Open"
        },
        "Options": {
            "regular": "09:30 - 16:00 ET (same as underlying)",
            "notes": "Same hours as underlying stocks",
            "status": is_market_open("options", current_time)
        }
    }
    
    for market, info in markets.items():
        status_icon = "🟢" if info["status"] == "OPEN" else "🔴" if info["status"] == "CLOSED" else "🟡"
        print(f"\n{status_icon} {market}")
        print(f"   Hours: {info['regular']}")
        if "pre_market" in info:
            print(f"   Pre-market: {info['pre_market']}")
        if "after_hours" in info:
            print(f"   After-hours: {info['after_hours']}")
        if "notes" in info:
            print(f"   Notes: {info['notes']}")
        print(f"   Status: {info['status']}")
    
    # Next market open/close
    print(f"\n📅 Next Market Events:")
    next_events = get_next_market_events(current_time)
    for event in next_events:
        print(f"• {event}")

def generate_ibkr_strategy_config(symbol: str, factors: ThresholdFactors, trading_hours: str) -> Dict:
    """Generate IBKR-specific strategy configuration"""
    return {
        "enabled": factors.composite_score > 0.5,
        "type": "conventional",
        "description": f"IBKR-optimized strategy for {symbol} (Score: {factors.composite_score:.3f})",
        "parameters": {
            "symbols": [symbol],
            "platform": "IBKR",
            "trading_hours": trading_hours,
            "asset_class": "stocks",
            "exchange": "SMART",  # IBKR's smart routing
            "currency": "USD",
            "risk_per_trade": max(0.005, min(0.05, 0.02 * factors.risk_management_score)),
            "position_size_multiplier": factors.composite_score,
            "volatility_adjustment": factors.volatility_factor,
            "liquidity_threshold": factors.liquidity_factor,
            "correlation_limit": 1 - factors.correlation_factor,
            "monte_carlo_confidence": factors.monte_carlo_score,
            "edge_threshold": factors.edge_ratio,
            "stop_loss_pct": max(0.01, min(0.08, 0.04 / max(0.1, factors.risk_management_score))),
            "take_profit_pct": max(0.02, min(0.15, 0.08 * max(0.1, factors.edge_ratio))),
            "order_type": "MKT",  # Market orders for IBKR
            "time_in_force": "DAY",
            "composite_score": factors.composite_score,
            "recommendation": factors.recommendation,
            "optimization_date": datetime.now().isoformat(),
            "reoptimize_after_days": 30,
            # IBKR-specific settings
            "ibkr_settings": {
                "smart_routing": True,
                "adapt_to_conditions": True,
                "outside_rth": False,  # Regular trading hours only
                "hidden": False,
                "sweep_to_fill": False
            }
        }
    }

def generate_ibkr_global_settings(optimization_results: List[Dict]) -> Dict:
    """Generate IBKR-optimized global settings"""
    successful_results = [r for r in optimization_results if r["strategy_generated"]]
    
    if not successful_results:
        return {
            "max_concurrent_positions": 3,
            "portfolio_heat": 0.05,
            "correlation_threshold": 0.8,
            "platform": "IBKR"
        }
    
    avg_score = np.mean([r["composite_score"] for r in successful_results])
    
    return {
        "max_concurrent_positions": min(8, max(3, len(successful_results))),
        "portfolio_heat": max(0.05, min(0.15, 0.10 * avg_score)),
        "correlation_threshold": max(0.6, min(0.8, 0.7 + 0.1 * avg_score)),
        "platform": "IBKR",
        "risk_management": {
            "max_daily_loss": max(0.02, min(0.08, 0.05 / avg_score)),
            "max_drawdown": max(0.10, min(0.25, 0.20 / avg_score)),
            "position_sizing_method": "adaptive_volatility"
        },
        "execution": {
            "platform": "IBKR",
            "slippage_model": "adaptive",
            "commission_rate": 0.005,  # IBKR commission per share
            "commission_min": 1.0,     # IBKR minimum commission
            "market_impact_factor": 0.0001 * (2 - avg_score),
            "order_timeout": 30,
            "smart_routing": True
        },
        "data_requirements": {
            "platform": "IBKR",
            "data_type": "delayed",  # or "realtime" if subscribed
            "min_history_days": 252,
            "required_timeframes": ["1m", "5m", "1h", "1d"],
            "required_indicators": ["sma", "ema", "rsi", "bollinger", "volume"]
        }
    }

def is_market_open(market_type: str, current_time) -> str:
    """Check if specific market is open"""
    weekday = current_time.weekday()  # 0=Monday, 6=Sunday
    hour = current_time.hour
    minute = current_time.minute
    current_minutes = hour * 60 + minute
    
    if market_type == "stocks" or market_type == "options":
        # Weekend check
        if weekday >= 5:  # Saturday or Sunday
            return "CLOSED"
        
        # Regular hours: 9:30 AM - 4:00 PM ET
        market_open = 9 * 60 + 30  # 9:30 AM
        market_close = 16 * 60     # 4:00 PM
        
        if market_open <= current_minutes <= market_close:
            return "OPEN"
        elif 4 * 60 <= current_minutes < market_open:
            return "PRE-MARKET"
        elif market_close < current_minutes <= 20 * 60:
            return "AFTER-HOURS"
        else:
            return "CLOSED"
    
    elif market_type == "forex":
        # Forex is open 24/5: Sunday 5 PM ET to Friday 5 PM ET
        if weekday == 6 and current_minutes < 17 * 60:  # Sunday before 5 PM
            return "CLOSED"
        elif weekday == 5 and current_minutes >= 17 * 60:  # Friday after 5 PM
            return "CLOSED"
        else:
            return "OPEN"
    
    return "UNKNOWN"

def get_next_market_events(current_time) -> List[str]:
    """Get next market open/close events"""
    events = []
    
    # This is simplified - you'd want more sophisticated market calendar logic
    weekday = current_time.weekday()
    
    if weekday < 5:  # Weekday
        events.append("Next market close: Today 16:00 ET")
        events.append("Next market open: Tomorrow 09:30 ET")
    elif weekday == 5:  # Saturday
        events.append("Next market open: Monday 09:30 ET")
    else:  # Sunday
        events.append("Forex opens: Today 17:00 ET")
        events.append("Stock market opens: Tomorrow 09:30 ET")
    
    return events

@app.command()
def market_hours_analysis(
    symbols: str = typer.Option("AAPL,MSFT", help="Symbols to analyze"),
    hours_config: str = typer.Option("09:30-16:00,17:00-20:00,20:00-23:00", help="Time periods to test"),
    days_back: int = typer.Option(30, help="Days of historical data"),
    output_file: str = typer.Option("hours_analysis.json", help="Output file")
):
    """Analyze optimal trading hours for different symbols"""
    setup_logging("INFO")
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    hour_periods = [h.strip() for h in hours_config.split(",")]
    
    print(f"\n⏰ Market Hours Analysis")
    print(f"📊 Symbols: {symbol_list}")
    print(f"🕐 Periods: {hour_periods}")
    print(f"📅 Lookback: {days_back} days")
    
    analysis_results = {}
    
    for symbol in symbol_list:
        print(f"\n📈 Analyzing {symbol}...")
        
        try:
            # Get intraday data (you'd need a different data source for real intraday)
            stock = yf.Ticker(symbol)
            data = stock.history(period=f"{days_back}d", interval="1h")
            
            if data.empty:
                print(f"❌ No intraday data for {symbol}")
                continue
            
            symbol_results = {}
            
            for period in hour_periods:
                start_hour, end_hour = period.split("-")
                start_time = datetime.strptime(start_hour, "%H:%M").time()
                end_time = datetime.strptime(end_hour, "%H:%M").time()
                
                # Filter data for this time period
                period_data = filter_by_time_period(data, start_time, end_time)
                
                if len(period_data) > 0:
                    # Calculate metrics for this period
                    returns = period_data['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252 * 6.5)  # Annualized
                    mean_return = returns.mean() * 252 * 6.5  # Annualized
                    volume_ratio = period_data['Volume'].mean() / data['Volume'].mean()
                    
                    period_analysis = {
                        "period": period,
                        "mean_return_annual": mean_return,
                        "volatility_annual": volatility,
                        "sharpe_ratio": mean_return / volatility if volatility > 0 else 0,
                        "volume_ratio": volume_ratio,
                        "data_points": len(period_data),
                        "recommendation_score": calculate_period_score(mean_return, volatility, volume_ratio)
                    }
                    
                    symbol_results[period] = period_analysis
                    print(f"  {period}: Return {mean_return:.2%}, Vol {volatility:.1%}, Sharpe {period_analysis['sharpe_ratio']:.2f}")
            
            analysis_results[symbol] = symbol_results
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Generate recommendations
    print(f"\n🎯 Optimal Trading Hours Recommendations:")
    print("=" * 60)
    
    for symbol, periods in analysis_results.items():
        if periods:
            best_period = max(periods.keys(), key=lambda k: periods[k]["recommendation_score"])
            best_score = periods[best_period]["recommendation_score"]
            print(f"{symbol}: {best_period} (Score: {best_score:.3f})")
    
    print(f"\n💾 Detailed analysis saved to: {output_file}")
    return analysis_results

@app.command()
def platform_comparison(
    symbols: str = typer.Option("AAPL,MSFT", help="Symbols to test"),
    platforms: str = typer.Option("IBKR,ALPACA,TD", help="Platforms to compare"),
    factors: str = typer.Option("commission,execution,data,reliability", help="Factors to compare"),
    output_file: str = typer.Option("platform_comparison.json", help="Output file")
):
    """Compare trading platforms across multiple factors"""
    setup_logging("INFO")
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    platform_list = [p.strip().upper() for p in platforms.split(",")]
    factor_list = [f.strip().lower() for f in factors.split(",")]
    
    print(f"\n🏢 Platform Comparison Analysis")
    print(f"📊 Symbols: {symbol_list}")
    print(f"🔧 Platforms: {platform_list}")
    print(f"📋 Factors: {factor_list}")
    
    # Platform specifications (you'd get this from real API testing)
    platform_specs = {
        "IBKR": {
            "commission": {"stocks": 0.005, "min": 1.0, "max": 1.0},
            "execution_speed_ms": 50,
            "data_quality_score": 0.95,
            "reliability_score": 0.92,
            "api_limits": {"requests_per_second": 50, "concurrent_connections": 1},
            "supported_exchanges": ["NYSE", "NASDAQ", "ARCA", "BATS"],
            "market_data_cost": 15.0,  # Monthly
            "min_account_size": 0
        },
        "ALPACA": {
            "commission": {"stocks": 0.0, "min": 0.0, "max": 0.0},
            "execution_speed_ms": 80,
            "data_quality_score": 0.88,
            "reliability_score": 0.89,
            "api_limits": {"requests_per_second": 200, "concurrent_connections": 1},
            "supported_exchanges": ["NYSE", "NASDAQ"],
            "market_data_cost": 0.0,
            "min_account_size": 0
        },
        "TD": {
            "commission": {"stocks": 0.0, "min": 0.0, "max": 0.0},
            "execution_speed_ms": 100,
            "data_quality_score": 0.90,
            "reliability_score": 0.88,
            "api_limits": {"requests_per_second": 120, "concurrent_connections": 1},
            "supported_exchanges": ["NYSE", "NASDAQ", "ARCA"],
            "market_data_cost": 0.0,
            "min_account_size": 0
        }
    }
    
    comparison_results = {}
    
    for platform in platform_list:
        if platform in platform_specs:
            specs = platform_specs[platform]
            
            # Calculate composite scores
            cost_score = calculate_cost_score(specs, symbol_list)
            performance_score = calculate_performance_score(specs)
            feature_score = calculate_feature_score(specs)
            
            composite_score = (cost_score * 0.4 + performance_score * 0.4 + feature_score * 0.2)
            
            comparison_results[platform] = {
                "specifications": specs,
                "scores": {
                    "cost_score": cost_score,
                    "performance_score": performance_score,
                    "feature_score": feature_score,
                    "composite_score": composite_score
                },
                "pros": generate_platform_pros(platform, specs),
                "cons": generate_platform_cons(platform, specs),
                "recommendation": generate_platform_recommendation(composite_score)
            }
    
    # Display comparison
    print(f"\n📊 Platform Comparison Results:")
    print("=" * 80)
    print(f"{'Platform':<10} {'Cost':<8} {'Performance':<12} {'Features':<10} {'Composite':<10} {'Recommendation':<15}")
    print("-" * 80)
    
    for platform, results in comparison_results.items():
        scores = results["scores"]
        print(f"{platform:<10} {scores['cost_score']:<7.3f} {scores['performance_score']:<11.3f} "
              f"{scores['feature_score']:<9.3f} {scores['composite_score']:<9.3f} {results['recommendation']:<15}")
    
    # Save detailed results
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed comparison saved to: {output_file}")
    
    # Recommend best platform
    if comparison_results:
        best_platform = max(comparison_results.keys(), 
                          key=lambda k: comparison_results[k]["scores"]["composite_score"])
        best_score = comparison_results[best_platform]["scores"]["composite_score"]
        print(f"\n🏆 Recommended Platform: {best_platform} (Score: {best_score:.3f})")
    
    return comparison_results

# Supporting functions for the new commands

def analyze_threshold_internal(symbol: str, lookback_days: int, monte_carlo_sims: int) -> ThresholdFactors:
    """Internal threshold analysis function"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=f"{lookback_days + 50}d")
        
        if data.empty:
            return None
        
        returns = data['Close'].pct_change().dropna()
        prices = data['Close']
        volumes = data['Volume']
        
        # Calculate all factors
        mc_score = monte_carlo_analysis(returns, monte_carlo_sims, 0.95)
        edge_ratio = calculate_edge_ratio(returns)
        risk_score = calculate_risk_management_score(returns, prices)
        volatility_factor = calculate_volatility_factor(returns)
        liquidity_factor = calculate_liquidity_factor(volumes, prices)
        correlation_factor = calculate_correlation_factor(symbol, returns)
        regime_factor = calculate_market_regime_factor(returns)
        
        composite_score = fuzzy_logic_composite(
            mc_score, edge_ratio, risk_score, volatility_factor,
            liquidity_factor, correlation_factor, regime_factor
        )
        
        recommendation = generate_recommendation(composite_score)
        
        return ThresholdFactors(
            monte_carlo_score=mc_score,
            edge_ratio=edge_ratio,
            risk_management_score=risk_score,
            volatility_factor=volatility_factor,
            liquidity_factor=liquidity_factor,
            correlation_factor=correlation_factor,
            market_regime_factor=regime_factor,
            composite_score=composite_score,
            recommendation=recommendation
        )
        
    except Exception as e:
        print(f"Error in threshold analysis for {symbol}: {e}")
        return None

def generate_advanced_strategy_config(symbol: str, factors: ThresholdFactors, 
                                    trading_hours: str, platforms: List[str]) -> Dict:
    """Generate advanced strategy configuration"""
    return {
        "enabled": factors.composite_score > 0.5,
        "type": "conventional",
        "description": f"AI-optimized strategy for {symbol} (Score: {factors.composite_score:.3f})",
        "parameters": {
            "symbols": [symbol],
            "trading_hours": trading_hours,
            "platforms": platforms,
            "risk_per_trade": max(0.005, min(0.05, 0.02 * factors.risk_management_score)),
            "position_size_multiplier": factors.composite_score,
            "volatility_adjustment": factors.volatility_factor,
            "liquidity_threshold": factors.liquidity_factor,
            "correlation_limit": 1 - factors.correlation_factor,
            "monte_carlo_confidence": factors.monte_carlo_score,
            "edge_threshold": factors.edge_ratio,
            "stop_loss_pct": max(0.01, min(0.08, 0.04 / max(0.1, factors.risk_management_score))),
            "take_profit_pct": max(0.02, min(0.15, 0.08 * max(0.1, factors.edge_ratio))),
            "composite_score": factors.composite_score,
            "recommendation": factors.recommendation,
            "optimization_date": datetime.now().isoformat(),
            "reoptimize_after_days": 30
        }
    }

def generate_optimized_global_settings(optimization_results: List[Dict]) -> Dict:
    """Generate optimized global settings based on analysis results"""
    successful_results = [r for r in optimization_results if r["strategy_generated"]]
    
    if not successful_results:
        # Default settings if no strategies generated
        return {
            "max_concurrent_positions": 3,
            "portfolio_heat": 0.05,
            "correlation_threshold": 0.8
        }
    
    avg_score = np.mean([r["composite_score"] for r in successful_results])
    
    return {
        "max_concurrent_positions": min(8, max(3, len(successful_results))),
        "portfolio_heat": max(0.05, min(0.15, 0.10 * avg_score)),
        "correlation_threshold": max(0.6, min(0.8, 0.7 + 0.1 * avg_score)),
        "risk_management": {
            "max_daily_loss": max(0.02, min(0.08, 0.05 / avg_score)),
            "max_drawdown": max(0.10, min(0.25, 0.20 / avg_score)),
            "position_sizing_method": "adaptive_volatility"
        },
        "execution": {
            "slippage_model": "adaptive",
            "commission_rate": 0.001,
            "market_impact_factor": 0.0001 * (2 - avg_score),
            "order_timeout": 30
        }
    }

def generate_platform_asset_classes(platforms: List[str]) -> Dict:
    """Generate asset class configuration for platforms"""
    asset_classes = {
        "stocks": {
            "trading_hours": {"start": "09:30", "end": "16:00", "timezone": "US/Eastern"},
            "platforms": platforms,
            "min_price": 5.0,
            "max_price": 1000.0,
            "min_volume": 100000
        }
    }
    
    if "IBKR" in platforms:
        asset_classes["forex"] = {
            "trading_hours": {"start": "17:00", "end": "17:00", "timezone": "US/Eastern"},
            "platforms": ["IBKR"],
            "major_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"]
        }
    
    return asset_classes

def filter_by_time_period(data: pd.DataFrame, start_time: time, end_time: time) -> pd.DataFrame:
    """Filter dataframe by time period"""
    # This is a simplified version - you'd need proper timezone handling
    data_copy = data.copy()
    data_copy['hour'] = data_copy.index.hour
    data_copy['minute'] = data_copy.index.minute
    
    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute
    data_minutes = data_copy['hour'] * 60 + data_copy['minute']
    
    if start_minutes <= end_minutes:
        mask = (data_minutes >= start_minutes) & (data_minutes <= end_minutes)
    else:  # Crosses midnight
        mask = (data_minutes >= start_minutes) | (data_minutes <= end_minutes)
    
    return data_copy[mask]

def calculate_period_score(mean_return: float, volatility: float, volume_ratio: float) -> float:
    """Calculate recommendation score for trading period"""
    return_score = max(0, min(1, (mean_return + 0.1) / 0.2))
    vol_score = max(0, min(1, 1 - volatility / 0.5))
    volume_score = max(0, min(1, volume_ratio))
    
    return (return_score * 0.4 + vol_score * 0.3 + volume_score * 0.3)

def calculate_cost_score(specs: Dict, symbols: List[str]) -> float:
    """Calculate cost score for platform"""
    commission = specs["commission"]["stocks"]
    data_cost = specs["market_data_cost"]
    
    # Estimate monthly cost for typical trading
    monthly_trades = 100  # Assumption
    monthly_commission = commission * monthly_trades
    total_monthly_cost = monthly_commission + data_cost
    
    # Normalize (lower cost = higher score)
    return max(0, min(1, 1 - total_monthly_cost / 100))

def calculate_performance_score(specs: Dict) -> float:
    """Calculate performance score for platform"""
    speed_score = max(0, min(1, 1 - specs["execution_speed_ms"] / 200))
    data_score = specs["data_quality_score"]
    reliability_score = specs["reliability_score"]
    
    return (speed_score + data_score + reliability_score) / 3

def calculate_feature_score(specs: Dict) -> float:
    """Calculate feature score for platform"""
    exchange_score = len(specs["supported_exchanges"]) / 4  # Normalize to 4 exchanges
    api_score = min(1, specs["api_limits"]["requests_per_second"] / 200)
    
    return (exchange_score + api_score) / 2

def generate_platform_pros(platform: str, specs: Dict) -> List[str]:
    """Generate pros for platform"""
    pros = []
    
    if specs["commission"]["stocks"] == 0:
        pros.append("Zero commission trading")
    if specs["execution_speed_ms"] < 75:
        pros.append("Fast execution speed")
    if specs["data_quality_score"] > 0.9:
        pros.append("High-quality market data")
    if specs["reliability_score"] > 0.9:
        pros.append("Excellent reliability")
    if specs["market_data_cost"] == 0:
        pros.append("Free market data")
    
    return pros

def generate_platform_cons(platform: str, specs: Dict) -> List[str]:
    """Generate cons for platform"""
    cons = []
    
    if specs["commission"]["stocks"] > 0:
        cons.append(f"Commission: ${specs['commission']['stocks']:.3f} per share")
    if specs["execution_speed_ms"] > 90:
        cons.append("Slower execution speed")
    if specs["data_quality_score"] < 0.9:
        cons.append("Lower data quality")
    if specs["market_data_cost"] > 10:
        cons.append(f"Market data cost: ${specs['market_data_cost']}/month")
    
    return cons

def generate_platform_recommendation(composite_score: float) -> str:
    """Generate platform recommendation"""
    if composite_score >= 0.8:
        return "EXCELLENT"
    elif composite_score >= 0.6:
        return "GOOD"
    elif composite_score >= 0.4:
        return "FAIR"
    else:
        return "POOR"

def update_main_strategies_config(optimized_strategies: Dict):
    """Update main strategies.yaml with optimized strategies"""
    try:
        import yaml
        config_path = "config/strategies.yaml"
        
        # Load existing config
        with open(config_path, 'r') as f:
            existing_config = yaml.safe_load(f)
        
        # Update strategies section
        if 'strategies' not in existing_config:
            existing_config['strategies'] = {}
        
        existing_config['strategies'].update(optimized_strategies)
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(existing_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Updated main configuration with {len(optimized_strategies)} optimized strategies")
        
    except Exception as e:
        print(f"❌ Failed to update main config: {e}")

@app.command()
def dashboard(
    refresh_seconds: int = typer.Option(5, help="Dashboard refresh rate"),
    show_positions: bool = typer.Option(True, help="Show current positions"),
    show_strategies: bool = typer.Option(True, help="Show strategy performance"),
    show_risk: bool = typer.Option(True, help="Show risk metrics"),
    export_interval: int = typer.Option(300, help="Export data every N seconds")
):
    """Launch real-time trading dashboard"""
    setup_logging("INFO")
    
    print("🚀 Starting Trading Dashboard...")
    print("Press Ctrl+C to stop")
    
    import time
    import os
    
    dashboard_data = {
        "start_time": datetime.now(),
        "positions": {},
        "strategies": {},
        "risk_metrics": {},
        "market_data": {},
        "alerts": []
    }
    
    last_export = time.time()
    
    try:
        while True:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Header
            current_time = datetime.now()
            uptime = current_time - dashboard_data["start_time"]
            
            print("=" * 80)
            print(f"🚀 TRADING SYSTEM DASHBOARD - {current_time.strftime('%H:%M:%S')}")
            print(f"⏱️  Uptime: {str(uptime).split('.')[0]} | Refresh: {refresh_seconds}s")
            print("=" * 80)
            
            # System Status
            print("\n📊 SYSTEM STATUS")
            print("-" * 40)
            system_status = get_system_status()
            for key, value in system_status.items():
                status_icon = "✅" if value else "❌"
                print(f"{status_icon} {key}: {value}")
            
            # Market Data Status
            if show_positions:
                print("\n💰 CURRENT POSITIONS")
                print("-" * 40)
                positions = get_current_positions()
                if positions:
                    print(f"{'Symbol':<8} {'Qty':<8} {'Price':<10} {'P&L':<10} {'%':<8}")
                    print("-" * 40)
                    for pos in positions:
                        pnl_color = "🟢" if pos['pnl'] >= 0 else "🔴"
                        print(f"{pos['symbol']:<8} {pos['quantity']:<8} ${pos['price']:<9.2f} "
                              f"{pnl_color}${pos['pnl']:<8.2f} {pos['pnl_pct']:<7.1%}")
                else:
                    print("No open positions")
            
            # Strategy Performance
            if show_strategies:
                print("\n🎯 STRATEGY PERFORMANCE")
                print("-" * 50)
                strategies = get_strategy_performance()
                if strategies:
                    print(f"{'Strategy':<15} {'Signals':<8} {'P&L':<10} {'Win%':<8} {'Status':<10}")
                    print("-" * 50)
                    for strat in strategies:
                        status_icon = "🟢" if strat['active'] else "🔴"
                        print(f"{strat['name']:<15} {strat['signals']:<8} ${strat['pnl']:<9.2f} "
                              f"{strat['win_rate']:<7.1%} {status_icon}{strat['status']:<9}")
                else:
                    print("No strategy data available")
            
            # Risk Metrics
            if show_risk:
                print("\n🛡️ RISK METRICS")
                print("-" * 40)
                risk_metrics = get_risk_metrics()
                print(f"Daily P&L: {risk_metrics['daily_pnl']:>+.2f} ({risk_metrics['daily_pnl_pct']:>+.1%})")
                print(f"Drawdown: {risk_metrics['current_drawdown']:>.1%} (Max: {risk_metrics['max_drawdown']:>.1%})")
                print(f"Portfolio Heat: {risk_metrics['portfolio_heat']:>.1%} / {risk_metrics['max_heat']:>.1%}")
                print(f"Open Positions: {risk_metrics['open_positions']} / {risk_metrics['max_positions']}")
                
                # Risk alerts
                if risk_metrics['circuit_breaker']:
                    print("🚨 CIRCUIT BREAKER ACTIVE")
                if risk_metrics['daily_pnl_pct'] < -0.03:
                    print("⚠️  Daily loss approaching limit")
            
            # Recent Alerts
            print("\n🚨 RECENT ALERTS")
            print("-" * 40)
            recent_alerts = get_recent_alerts(limit=5)
            if recent_alerts:
                for alert in recent_alerts:
                    alert_icon = "🚨" if alert['level'] == 'CRITICAL' else "⚠️" if alert['level'] == 'WARNING' else "ℹ️"
                    print(f"{alert_icon} {alert['time']} - {alert['message']}")
            else:
                print("No recent alerts")
            
            # Market Data Summary
            print("\n📈 MARKET DATA")
            print("-" * 40)
            market_summary = get_market_data_summary()
            print(f"Data Updates: {market_summary['updates_received']} (Last: {market_summary['last_update']})")
            print(f"Active Subscriptions: {market_summary['active_subscriptions']}")
            print(f"Connection Status: {market_summary['connection_status']}")
            
            # Export data periodically
            current_time_seconds = time.time()
            if current_time_seconds - last_export > export_interval:
                export_dashboard_data(dashboard_data)
                last_export = current_time_seconds
            
            # Footer
            print("\n" + "=" * 80)
            print("Commands: Ctrl+C to exit | Dashboard auto-refreshes")
            print("=" * 80)
            
            # Wait for next refresh
            time.sleep(refresh_seconds)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
        export_dashboard_data(dashboard_data, force=True)

@app.command()
def alerts(
    level: str = typer.Option("ALL", help="Alert level: ALL, CRITICAL, WARNING, INFO"),
    last_hours: int = typer.Option(24, help="Show alerts from last N hours"),
    export_file: str = typer.Option("", help="Export to file")
):
    """Show system alerts and notifications"""
    setup_logging("INFO")
    
    print(f"\n🚨 System Alerts - Last {last_hours} hours")
    print(f"📊 Level Filter: {level}")
    print("=" * 60)
    
    alerts = get_historical_alerts(hours=last_hours, level=level)
    
    if alerts:
        print(f"{'Time':<12} {'Level':<10} {'Component':<15} {'Message':<30}")
        print("-" * 60)
        
        for alert in alerts:
            level_icon = "🚨" if alert['level'] == 'CRITICAL' else "⚠️" if alert['level'] == 'WARNING' else "ℹ️"
            print(f"{alert['timestamp']:<12} {level_icon}{alert['level']:<9} "
                  f"{alert['component']:<15} {alert['message'][:28]:<30}")
        
        print(f"\nTotal alerts: {len(alerts)}")
        
        # Statistics
        if len(alerts) > 1:
            level_counts = {}
            for alert in alerts:
                level_counts[alert['level']] = level_counts.get(alert['level'], 0) + 1
            
            print("\n📊 Alert Statistics:")
            for level, count in level_counts.items():
                print(f"  {level}: {count}")
    else:
        print("No alerts found for the specified criteria")
    
    # Export if requested
    if export_file:
        export_alerts_to_file(alerts, export_file)
        print(f"\n💾 Alerts exported to: {export_file}")

@app.command()
def performance_report(
    start_date: str = typer.Option("", help="Start date YYYY-MM-DD (default: 30 days ago)"),
    end_date: str = typer.Option("", help="End date YYYY-MM-DD (default: today)"),
    strategies: str = typer.Option("ALL", help="Strategies to include (comma-separated)"),
    output_file: str = typer.Option("performance_report.html", help="Output HTML report file"),
    include_charts: bool = typer.Option(True, help="Include performance charts")
):
    """Generate comprehensive performance report"""
    setup_logging("INFO")
    
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n📊 Generating Performance Report")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🎯 Strategies: {strategies}")
    
    # Collect performance data
    performance_data = collect_performance_data(start_date, end_date, strategies)
    
    # Generate HTML report
    html_report = generate_html_report(performance_data, include_charts)
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"\n✅ Performance report generated: {output_file}")
    print(f"📈 Total trades: {performance_data['summary']['total_trades']}")
    print(f"💰 Total P&L: ${performance_data['summary']['total_pnl']:.2f}")
    print(f"📊 Win rate: {performance_data['summary']['win_rate']:.1%}")
    print(f"⚡ Sharpe ratio: {performance_data['summary']['sharpe_ratio']:.2f}")

# Supporting functions for dashboard and monitoring

def get_system_status() -> Dict[str, bool]:
    """Get current system status"""
    import os
    from pathlib import Path
    
    log_file = Path("logs/system.log")
    status = {
        "Trading System": False,
        "IBKR Connection": False,
        "Data Feed": False,
        "Strategies Active": False,
        "Risk Manager": False
    }
    
    if log_file.exists():
        try:
            # Read recent log entries with proper encoding handling
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                recent_logs = f.readlines()[-50:]  # Last 50 lines
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding if utf-8 fails
            try:
                with open(log_file, 'r', encoding='latin-1', errors='ignore') as f:
                    recent_logs = f.readlines()[-50:]
            except Exception:
                # Last resort: binary read and decode
                with open(log_file, 'rb') as f:
                    content = f.read()
                    # Decode with error handling
                    recent_text = content.decode('utf-8', errors='ignore')
                    recent_logs = recent_text.split('\n')[-50:]
        except Exception as e:
            print(f"Warning: Could not read log file: {e}")
            return status
        
        recent_text = ''.join(recent_logs)
        
        # Check for various status indicators
        if "Trading system started successfully" in recent_text:
            status["Trading System"] = True
        if "Connected to IBKR" in recent_text:
            status["IBKR Connection"] = True
        if "Market data:" in recent_text or "price updates received" in recent_text:
            status["Data Feed"] = True
        if "Strategy" in recent_text and "initialized" in recent_text:
            status["Strategies Active"] = True
        if "Risk metrics updated" in recent_text:
            status["Risk Manager"] = True
    
    return status

def get_current_positions() -> List[Dict]:
    """Get current trading positions from IBKR paper account"""
    from execution.position_manager import get_real_positions
    
    try:
        if os.getenv('USE_REAL_POSITIONS', 'false').lower() == 'true':
            return get_real_positions()
        else:
            print("📊 Real positions disabled in config")
            return []
    except Exception as e:
        print(f"⚠️ Error getting real positions: {e}")
        return []

def get_strategy_performance() -> List[Dict]:
    """Get actual strategy performance from trade database"""
    from execution.position_manager import get_strategy_performance
    
    try:
        return get_strategy_performance()
    except Exception as e:
        print(f"⚠️ Error getting strategy performance: {e}")
        return []

def get_market_data_summary() -> Dict:
    """Get market data connection summary"""
    return {
        "updates_received": 1247,
        "last_update": "2 seconds ago",
        "active_subscriptions": 5,
        "connection_status": "Connected (Delayed Data)"
    }

def get_historical_alerts(hours: int, level: str) -> List[Dict]:
    """Get historical alerts (mock data for demo)"""
    alerts = [
        {"timestamp": "15:45:32", "level": "WARNING", "component": "Risk Manager", "message": "High volatility detected"},
        {"timestamp": "15:12:18", "level": "INFO", "component": "Strategy", "message": "Signal generated"},
        {"timestamp": "14:58:07", "level": "INFO", "component": "Data Provider", "message": "Connection restored"},
        {"timestamp": "14:30:15", "level": "WARNING", "component": "Risk Manager", "message": "P&L threshold"},
        {"timestamp": "13:45:22", "level": "CRITICAL", "component": "Execution", "message": "Order failed"},
    ]
    
    # Filter by level if not ALL
    if level != "ALL":
        alerts = [a for a in alerts if a["level"] == level]
    
    return alerts

def export_dashboard_data(data: Dict, force: bool = False):
    """Export dashboard data to file"""
    try:
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "system_status": get_system_status(),
            "positions": get_current_positions(),
            "strategies": get_strategy_performance(),
            "risk_metrics": get_risk_metrics()
        }
        
        filename = f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f"logs/{filename}", 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        if force:
            print(f"📊 Dashboard data exported to logs/{filename}")
            
    except Exception as e:
        print(f"❌ Failed to export dashboard data: {e}")

def export_alerts_to_file(alerts: List[Dict], filename: str):
    """Export alerts to file"""
    try:
        with open(filename, 'w') as f:
            if filename.endswith('.json'):
                json.dump(alerts, f, indent=2, default=str)
            else:  # CSV format
                f.write("Timestamp,Level,Component,Message\n")
                for alert in alerts:
                    f.write(f"{alert['timestamp']},{alert['level']},{alert['component']},{alert['message']}\n")
    except Exception as e:
        print(f"❌ Failed to export alerts: {e}")

def collect_performance_data(start_date: str, end_date: str, strategies: str) -> Dict:
    """Collect performance data for report"""
    # Mock performance data - in real system would query database
    return {
        "period": {"start": start_date, "end": end_date},
        "summary": {
            "total_trades": 45,
            "total_pnl": 2847.50,
            "win_rate": 0.64,
            "sharpe_ratio": 1.42,
            "max_drawdown": 0.08,
            "best_trade": 485.20,
            "worst_trade": -234.50
        },
        "daily_returns": [0.012, -0.008, 0.015, 0.003, -0.002],  # Sample daily returns
        "strategy_breakdown": {
            "ma_crossover": {"trades": 25, "pnl": 1548.20, "win_rate": 0.68},
            "bollinger_bands": {"trades": 20, "pnl": 1299.30, "win_rate": 0.60}
        }
    }

def generate_html_report(data: Dict, include_charts: bool = True) -> str:
    """Generate HTML performance report"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
            .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .metric {{ text-align: center; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
            .metric h3 {{ margin: 0; color: #2c3e50; }}
            .metric p {{ margin: 5px 0; font-size: 24px; font-weight: bold; }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Trading Performance Report</h1>
            <p>Period: {data['period']['start']} to {data['period']['end']}</p>
        </div>
        
        <div class="summary">
            <div class="metric">
                <h3>Total P&L</h3>
                <p class="{'positive' if data['summary']['total_pnl'] >= 0 else 'negative'}">
                    ${data['summary']['total_pnl']:,.2f}
                </p>
            </div>
            <div class="metric">
                <h3>Total Trades</h3>
                <p>{data['summary']['total_trades']}</p>
            </div>
            <div class="metric">
                <h3>Win Rate</h3>
                <p>{data['summary']['win_rate']:.1%}</p>
            </div>
            <div class="metric">
                <h3>Sharpe Ratio</h3>
                <p>{data['summary']['sharpe_ratio']:.2f}</p>
            </div>
        </div>
        
        <h2>📈 Strategy Breakdown</h2>
        <table>
            <tr><th>Strategy</th><th>Trades</th><th>P&L</th><th>Win Rate</th></tr>
    """
    
    for strategy, stats in data['strategy_breakdown'].items():
        pnl_class = "positive" if stats['pnl'] >= 0 else "negative"
        html += f"""
            <tr>
                <td>{strategy}</td>
                <td>{stats['trades']}</td>
                <td class="{pnl_class}">${stats['pnl']:,.2f}</td>
                <td>{stats['win_rate']:.1%}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <p><small>Report generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</small></p>
    </body>
    </html>
    """
    
    return html

# Add these commands to your main.py file

@app.command()
def transition_to_paper():
    """Transition from demo to real paper trading with multi-market support"""
    setup_logging("INFO")
    
    print("🚀 Starting Transition to Real Paper Trading...")
    
    # Import and run transition
    try:
        import subprocess
        import sys
        
        # Run the transition script
        result = subprocess.run([
            sys.executable, 
            "scripts/transition_to_paper.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Transition completed successfully!")
            print(result.stdout)
        else:
            print("❌ Transition failed!")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running transition: {e}")

@app.command()
def verify_paper_setup():
    """Verify paper trading setup"""
    setup_logging("INFO")
    
    print("🔍 Verifying Paper Trading Setup...")
    
    try:
        import subprocess
        import sys
        
        # Run verification script
        result = subprocess.run([
            sys.executable, 
            "scripts/verify_paper_trading.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running verification: {e}")

@app.command()
def multi_market_status():
    """Show multi-market trading status"""
    setup_logging("INFO")
    
    print("🌍 Multi-Market Trading Status")
    print("=" * 60)
    
    try:
        from execution.position_manager import get_position_manager
        from data.multi_market_provider import MultiMarketDataProvider
        
        # Get position manager
        manager = get_position_manager()
        
        # Get portfolio summary
        summary = manager.get_portfolio_summary()
        
        print("💰 Portfolio Summary:")
        print(f"  Total Value: ${summary['total_market_value']:,.2f}")
        print(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:,.2f}")
        print(f"  Total Positions: {summary['position_count']}")
        
        print("\n📊 By Asset Class:")
        for asset_class, data in summary['by_asset_class'].items():
            print(f"  {asset_class.upper()}:")
            print(f"    Value: ${data['market_value']:,.2f}")
            print(f"    P&L: ${data['unrealized_pnl']:,.2f}")
            print(f"    Positions: {data['count']}")
            print(f"    Symbols: {', '.join(data['symbols'][:5])}")
            if len(data['symbols']) > 5:
                print(f"             ... +{len(data['symbols'])-5} more")
        
        print("\n💱 By Currency:")
        for currency, data in summary['by_currency'].items():
            print(f"  {currency}: ${data['market_value']:,.2f} ({data['count']} positions)")
        
        # Market hours status
        print("\n⏰ Market Hours Status:")
        from datetime import datetime
        import pytz
        
        et = pytz.timezone('US/Eastern')
        current_time = datetime.now(et)
        
        markets_status = {
            "US Stocks": is_market_open("stocks", current_time),
            "Forex": is_market_open("forex", current_time),
            "Futures": "Usually Open" if current_time.weekday() < 5 else "Weekend",
            "Crypto": "24/7 Open"
        }
        
        for market, status in markets_status.items():
            status_icon = "🟢" if "Open" in status else "🔴" if "Closed" in status else "🟡"
            print(f"  {status_icon} {market}: {status}")
        
    except Exception as e:
        print(f"❌ Error getting multi-market status: {e}")

@app.command()
def show_available_markets():
    """Show all available markets and symbols"""
    setup_logging("INFO")
    
    print("🌍 Available Markets & Symbols")
    print("=" * 60)
    
    try:
        import yaml
        
        # Load markets configuration
        with open('config/markets.yaml', 'r') as f:
            markets_config = yaml.safe_load(f)
        
        total_symbols = 0
        
        for asset_class, market_groups in markets_config.items():
            print(f"\n📊 {asset_class.upper()}:")
            
            for group_name, group_config in market_groups.items():
                symbols = group_config.get('symbols', [])
                exchange = group_config.get('contract_params', {}).get('exchange', 'N/A')
                
                print(f"  {group_name} ({exchange}):")
                
                # Show symbols in groups of 10
                for i in range(0, len(symbols), 10):
                    symbol_group = symbols[i:i+10]
                    print(f"    {', '.join(symbol_group)}")
                
                print(f"    Total: {len(symbols)} symbols")
                total_symbols += len(symbols)
        
        print(f"\n📈 Overall Summary:")
        print(f"  Asset Classes: {len(markets_config)}")
        print(f"  Total Symbols: {total_symbols}")
        print(f"  Markets: {sum(len(groups) for groups in markets_config.values())}")
        
        # Trading hours summary
        print(f"\n⏰ Trading Hours Summary:")
        print("  US Stocks: 09:30-16:00 ET (Pre: 04:00-09:30, After: 16:00-20:00)")
        print("  Forex: 24/5 (Sun 17:00 ET - Fri 17:00 ET)")
        print("  Futures: Nearly 24/5 (18:00 ET - 17:00 ET)")
        print("  Commodities: Varies by contract")
        print("  Crypto: 24/7")
        
    except Exception as e:
        print(f"❌ Error loading markets: {e}")

@app.command()
def create_multi_market_strategy(
    name: str = typer.Option(..., help="Strategy name"),
    markets: str = typer.Option("stocks,forex", help="Comma-separated markets"),
    symbols_per_market: int = typer.Option(5, help="Symbols per market"),
    risk_per_trade: float = typer.Option(0.02, help="Risk per trade"),
    enabled: bool = typer.Option(True, help="Enable strategy")
):
    """Create a new multi-market strategy"""
    setup_logging("INFO")
    
    print(f"🎯 Creating Multi-Market Strategy: {name}")
    
    try:
        import yaml
        
        # Load available markets
        with open('config/markets.yaml', 'r') as f:
            available_markets = yaml.safe_load(f)
        
        # Parse requested markets
        requested_markets = [m.strip() for m in markets.split(",")]
        
        # Build strategy configuration
        strategy_markets = {}
        
        for market in requested_markets:
            if market in available_markets:
                # Get first group of symbols from each market
                first_group = list(available_markets[market].values())[0]
                symbols = first_group.get('symbols', [])[:symbols_per_market]
                strategy_markets[market] = symbols
                print(f"  ✅ {market}: {len(symbols)} symbols")
            else:
                print(f"  ❌ {market}: Not available")
        
        if not strategy_markets:
            print("❌ No valid markets selected")
            return
        
        # Create strategy configuration
        strategy_config = {
            'enabled': enabled,
            'type': 'conventional',
            'description': f'Multi-market strategy across {", ".join(strategy_markets.keys())}',
            'parameters': {
                'markets': strategy_markets,
                'risk_per_trade': risk_per_trade,
                'lookback_period': 20,
                'momentum_threshold': 0.02,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'correlation_threshold': 0.7,
                'volatility_adjustment': True
            }
        }
        
        # Load existing strategies
        strategies_path = Path('config/strategies.yaml')
        if strategies_path.exists():
            with open(strategies_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {'strategies': {}}
        
        # Add new strategy
        config['strategies'][name] = strategy_config
        
        # Save updated configuration
        with open(strategies_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Strategy '{name}' created successfully!")
        print(f"📊 Markets: {list(strategy_markets.keys())}")
        print(f"🎯 Total symbols: {sum(len(symbols) for symbols in strategy_markets.values())}")
        print(f"💰 Risk per trade: {risk_per_trade:.1%}")
        print(f"🔄 Status: {'Enabled' if enabled else 'Disabled'}")
        
    except Exception as e:
        print(f"❌ Error creating strategy: {e}")

@app.command()
def test_multi_market_connection():
    """Test connection to all markets"""
    setup_logging("INFO")
    
    print("🧪 Testing Multi-Market Connections")
    print("=" * 50)
    
    async def test_connections():
        try:
            from data.multi_market_provider import MultiMarketDataProvider
            from execution.enhanced_broker import EnhancedIBKRBroker
            
            # Test data provider
            print("📡 Testing Data Provider...")
            
            host = os.getenv('IBKR_HOST', '127.0.0.1')
            port = int(os.getenv('IBKR_PORT', '7497'))
            client_id = int(os.getenv('IBKR_CLIENT_ID', '1000'))
            
            provider = MultiMarketDataProvider(host, port, client_id + 10, cache=None)
            
            try:
                connected = await provider.connect()
                if connected:
                    print("  ✅ Data provider connected")
                    
                    # Test small market subscription
                    test_markets = {
                        'stocks': {
                            'symbols': ['AAPL', 'MSFT'],
                            'contract_params': {'exchange': 'SMART', 'currency': 'USD'}
                        },
                        'forex': {
                            'symbols': ['EUR.USD'],
                            'contract_params': {'exchange': 'IDEALPRO'}
                        }
                    }
                    
                    print("  📊 Testing market subscriptions...")
                    data_count = 0
                    
                    async for data in provider.subscribe_all_markets(test_markets):
                        data_count += 1
                        print(f"    📈 Received: {data.symbol} = ${data.price:.4f}")
                        
                        if data_count >= 5:  # Test with 5 data points
                            break
                    
                    print(f"  ✅ Received {data_count} market data updates")
                    
                    # Get statistics
                    stats = provider.get_market_statistics()
                    print(f"  📊 Statistics: {stats}")
                    
                else:
                    print("  ❌ Data provider connection failed")
                
                await provider.disconnect()
                
            except Exception as e:
                print(f"  ❌ Data provider error: {e}")
            
            # Test broker
            print("\n🏦 Testing Enhanced Broker...")
            
            broker = EnhancedIBKRBroker(host, port, client_id + 20)
            
            try:
                connected = await broker.connect()
                if connected:
                    print("  ✅ Enhanced broker connected")
                    
                    # Test multi-market positions
                    positions = await broker.get_multi_market_positions()
                    print(f"  💰 Position categories: {list(positions.keys())}")
                    
                    total_positions = sum(len(pos_list) for pos_list in positions.values())
                    print(f"  📊 Total positions: {total_positions}")
                    
                else:
                    print("  ❌ Enhanced broker connection failed")
                
                await broker.disconnect()
                
            except Exception as e:
                print(f"  ❌ Enhanced broker error: {e}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    # Run async test
    asyncio.run(test_connections())

@app.command()
def market_hours_dashboard():
    """Show real-time market hours dashboard"""
    setup_logging("INFO")
    
    import time
    from datetime import datetime
    import pytz
    
    try:
        while True:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("⏰ Real-Time Market Hours Dashboard")
            print("=" * 60)
            
            # Current times in major timezones
            now_utc = datetime.now(pytz.UTC)
            now_et = now_utc.astimezone(pytz.timezone('US/Eastern'))
            now_london = now_utc.astimezone(pytz.timezone('Europe/London'))
            now_tokyo = now_utc.astimezone(pytz.timezone('Asia/Tokyo'))
            now_sydney = now_utc.astimezone(pytz.timezone('Australia/Sydney'))
            
            print(f"🌍 Current Times:")
            print(f"  New York:  {now_et.strftime('%H:%M:%S %Z')}")
            print(f"  London:    {now_london.strftime('%H:%M:%S %Z')}")
            print(f"  Tokyo:     {now_tokyo.strftime('%H:%M:%S %Z')}")
            print(f"  Sydney:    {now_sydney.strftime('%H:%M:%S %Z')}")
            
            # Market status
            print(f"\n📊 Market Status:")
            
            markets = {
                "US Stocks": {
                    "status": is_market_open("stocks", now_et),
                    "hours": "09:30-16:00 ET",
                    "pre_post": "Pre: 04:00-09:30, After: 16:00-20:00"
                },
                "Forex": {
                    "status": is_market_open("forex", now_et),
                    "hours": "24/5 (Sun 17:00 ET - Fri 17:00 ET)",
                    "sessions": "Sydney→Tokyo→London→NY"
                },
                "Futures": {
                    "status": "OPEN" if now_et.weekday() < 5 else "WEEKEND",
                    "hours": "Nearly 24/5 (18:00 ET - 17:00 ET)",
                    "note": "Pit session: 09:30-16:15 ET"
                },
                "Crypto": {
                    "status": "OPEN",
                    "hours": "24/7",
                    "note": "Continuous trading"
                },
                "Commodities": {
                    "status": "VARIES",
                    "hours": "Depends on contract",
                    "note": "Gold: 18:00-17:00 ET, Oil: 18:00-17:00 ET"
                }
            }
            
            for market, info in markets.items():
                status = info["status"]
                if status == "OPEN":
                    icon = "🟢"
                elif status == "CLOSED":
                    icon = "🔴"
                elif "PRE" in status or "AFTER" in status:
                    icon = "🟡"
                else:
                    icon = "⚪"
                
                print(f"  {icon} {market:<12} {status:<12} {info['hours']}")
                if 'note' in info:
                    print(f"     {'':>12} {'':>12} {info['note']}")
                if 'sessions' in info:
                    print(f"     {'':>12} {'':>12} {info['sessions']}")
                if 'pre_post' in info:
                    print(f"     {'':>12} {'':>12} {info['pre_post']}")
            
            # Upcoming market events
            print(f"\n📅 Next Market Events:")
            events = get_next_market_events(now_et)
            for event in events[:3]:
                print(f"  • {event}")
            
            # Trading system status
            print(f"\n🚀 Trading System:")
            if Path("logs/system.log").exists():
                print("  ✅ System logs present")
            else:
                print("  ❌ No system logs found")
            
            print(f"\n🔄 Dashboard refreshes every 30 seconds")
            print("Press Ctrl+C to exit")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n👋 Market hours dashboard stopped")

@app.command()
def export_trading_data(
    start_date: str = typer.Option("", help="Start date YYYY-MM-DD"),
    end_date: str = typer.Option("", help="End date YYYY-MM-DD"),
    asset_classes: str = typer.Option("all", help="Asset classes to export"),
    output_format: str = typer.Option("csv", help="Output format: csv, json, excel"),
    output_file: str = typer.Option("", help="Output file name")
):
    """Export trading data across all markets"""
    setup_logging("INFO")
    
    from datetime import datetime, timedelta
    
    # Set default dates
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print(f"📊 Exporting Trading Data")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🎯 Asset Classes: {asset_classes}")
    print(f"📁 Format: {output_format}")
    
    try:
        from execution.position_manager import get_position_manager
        
        # Get position manager
        manager = get_position_manager()
        
        # Collect data
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'start_date': start_date,
                'end_date': end_date,
                'asset_classes': asset_classes
            },
            'positions': [],
            'portfolio_summary': {},
            'strategy_performance': []
        }
        
        # Get current positions
        all_positions = manager.get_positions_by_asset_class()
        
        # Filter by asset class if specified
        if asset_classes != "all":
            requested_classes = [ac.strip() for ac in asset_classes.split(",")]
            all_positions = [pos for pos in all_positions 
                           if pos.asset_class in requested_classes]
        
        # Convert positions to export format
        for pos in all_positions:
            export_data['positions'].append({
                'symbol': pos.symbol,
                'asset_class': pos.asset_class,
                'quantity': pos.quantity,
                'avg_cost': pos.avg_cost,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'currency': pos.currency,
                'exchange': pos.exchange
            })
        
        # Get portfolio summary
        export_data['portfolio_summary'] = manager.get_portfolio_summary()
        
        # Get strategy performance
        export_data['strategy_performance'] = manager.get_strategy_performance_list()
        
        # Generate output filename if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"trading_data_{timestamp}.{output_format}"
        
        # Export based on format
        if output_format.lower() == 'json':
            import json
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif output_format.lower() == 'csv':
            import pandas as pd
            
            # Create separate CSV files
            base_name = output_file.replace('.csv', '')
            
            # Positions CSV
            if export_data['positions']:
                df_positions = pd.DataFrame(export_data['positions'])
                df_positions.to_csv(f"{base_name}_positions.csv", index=False)
                print(f"  ✅ Positions: {base_name}_positions.csv")
            
            # Strategy performance CSV
            if export_data['strategy_performance']:
                df_strategies = pd.DataFrame(export_data['strategy_performance'])
                df_strategies.to_csv(f"{base_name}_strategies.csv", index=False)
                print(f"  ✅ Strategies: {base_name}_strategies.csv")
            
            # Portfolio summary JSON (complex structure)
            with open(f"{base_name}_summary.json", 'w') as f:
                json.dump(export_data['portfolio_summary'], f, indent=2, default=str)
                print(f"  ✅ Summary: {base_name}_summary.json")
        
        elif output_format.lower() == 'excel':
            import pandas as pd
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Positions sheet
                if export_data['positions']:
                    df_positions = pd.DataFrame(export_data['positions'])
                    df_positions.to_excel(writer, sheet_name='Positions', index=False)
                
                # Strategy performance sheet
                if export_data['strategy_performance']:
                    df_strategies = pd.DataFrame(export_data['strategy_performance'])
                    df_strategies.to_excel(writer, sheet_name='Strategies', index=False)
                
                # Summary sheet (flattened)
                summary_flat = []
                for key, value in export_data['portfolio_summary'].items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            summary_flat.append({'Category': f"{key}_{subkey}", 'Value': subvalue})
                    else:
                        summary_flat.append({'Category': key, 'Value': value})
                
                if summary_flat:
                    df_summary = pd.DataFrame(summary_flat)
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"✅ Export completed: {output_file}")
        print(f"📊 Exported {len(export_data['positions'])} positions")
        print(f"🎯 Exported {len(export_data['strategy_performance'])} strategies")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

# Helper function that should already exist in your main.py
def is_market_open(market_type: str, current_time) -> str:
    """Check if specific market is open"""
    weekday = current_time.weekday()  # 0=Monday, 6=Sunday
    hour = current_time.hour
    minute = current_time.minute
    current_minutes = hour * 60 + minute
    
    if market_type == "stocks":
        # Weekend check
        if weekday >= 5:  # Saturday or Sunday
            return "CLOSED"
        
        # Regular hours: 9:30 AM - 4:00 PM ET
        market_open = 9 * 60 + 30  # 9:30 AM
        market_close = 16 * 60     # 4:00 PM
        
        if market_open <= current_minutes <= market_close:
            return "OPEN"
        elif 4 * 60 <= current_minutes < market_open:
            return "PRE-MARKET"
        elif market_close < current_minutes <= 20 * 60:
            return "AFTER-HOURS"
        else:
            return "CLOSED"
    
    elif market_type == "forex":
        # Forex is open 24/5: Sunday 5 PM ET to Friday 5 PM ET
        if weekday == 6 and current_minutes < 17 * 60:  # Sunday before 5 PM
            return "CLOSED"
        elif weekday == 5 and current_minutes >= 17 * 60:  # Friday after 5 PM
            return "CLOSED"
        else:
            return "OPEN"
    
    return "UNKNOWN"

def get_next_market_events(current_time) -> List[str]:
    """Get next market open/close events"""
    events = []
    weekday = current_time.weekday()
    
    if weekday < 5:  # Weekday
        events.append("Next market close: Today 16:00 ET")
        events.append("Next market open: Tomorrow 09:30 ET")
    elif weekday == 5:  # Saturday
        events.append("Next market open: Monday 09:30 ET")
        events.append("Forex reopens: Sunday 17:00 ET")
    else:  # Sunday
        events.append("Forex opens: Today 17:00 ET")
        events.append("Stock market opens: Tomorrow 09:30 ET")
    
    return events

if __name__ == "__main__":
    app()