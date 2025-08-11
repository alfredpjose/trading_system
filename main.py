# main.py
import asyncio
import signal
import sys
from pathlib import Path
from typing import Dict, List
import typer
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Better approach - import from the package
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
        
        # Core components
        self.event_bus = EventBus()
        self.cache = DataCache(config.data.redis_url, config.data.cache_ttl)
        self.database = DatabaseManager(config.data.database_url)
        
        # Trading components
        self.data_provider = None
        self.broker = None
        self.risk_manager = None
        self.execution_engine = None
        
        # Strategies
        self.strategies = []
        
        # Monitoring
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing trading system...")
        
        # Setup monitoring first
        await self._setup_monitoring()
        
        # Initialize data layer
        await self.cache.connect()
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
        # Data provider
        self.data_provider = IBKRDataProvider(
            self.config.ibkr.host,
            self.config.ibkr.port,
            self.config.ibkr.client_id,
            self.cache
        )
        
        # Broker
        self.broker = IBKRBroker(
            self.config.ibkr.host,
            self.config.ibkr.port,
            self.config.ibkr.client_id + 1  # Different client ID
        )
        
        # Risk manager
        self.risk_manager = RiskManager(self.config.risk, 100000.0)  # $100k initial capital
        
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
                # Extract strategy type from name (e.g., "momentum_basic" -> "momentum")
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
        
        # Stop components
        await self.execution_engine.stop()
        await self.data_provider.disconnect()
        
        # Cleanup strategies
        for strategy in self.strategies:
            await strategy.cleanup()
        
        # Stop monitoring
        await self.metrics_collector.stop_collection()
        
        # Stop event bus
        await self.event_bus.stop()
        
        # Cleanup data layer
        await self.cache.disconnect()
        await self.database.cleanup()
        
        logger.info("Trading system stopped")

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
    
    # Check IBKR connection
    config = get_config()
    
    async def check_connections():
        try:
            # Test cache connection
            cache = DataCache(config.data.redis_url)
            await cache.connect()
            logger.info("✓ Redis cache connection OK")
            await cache.disconnect()
        except Exception as e:
            logger.error(f"✗ Redis cache connection failed: {e}")
        
        try:
            # Test database connection
            db = DatabaseManager(config.data.database_url)
            await db.initialize()
            logger.info("✓ Database connection OK")
            await db.cleanup()
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
        
        # Load and validate strategies
        try:
            strategy_configs = load_strategy_configs()
            enabled_strategies = [s for s in strategy_configs if s.enabled]
            logger.info(f"✓ Found {len(enabled_strategies)} enabled strategies")
            for strategy in enabled_strategies:
                logger.info(f"  - {strategy.name} (allocation: {strategy.capital_allocation:.1%})")
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

if __name__ == "__main__":
    app()