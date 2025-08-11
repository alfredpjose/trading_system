# strategies/base.py
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np
from loguru import logger

from core.interfaces import IStrategy, IEventBus, MarketData, Signal, Event, EventType, OrderAction

class BaseStrategy(IStrategy):
    """Base strategy implementation with common functionality"""
    
    def __init__(self, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]):
        self._strategy_id = strategy_id
        self.event_bus = event_bus
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # Data storage
        self.price_history: Dict[str, deque] = {}
        self.signals_generated = 0
        self.last_signals: Dict[str, Signal] = {}
        
        # Strategy parameters
        self.symbols = config.get('symbols', [])
        self.lookback_period = config.get('lookback_period', 20)
        self.min_confidence = config.get('min_confidence', 0.5)
        
    @property
    def strategy_id(self) -> str:
        return self._strategy_id
    
    async def initialize(self, config: Dict[str, Any]):
        """Initialize strategy with configuration"""
        self.config.update(config)
        logger.info(f"Strategy {self.strategy_id} initialized with config: {config}")
    
    async def process_data(self, data: MarketData) -> Optional[Signal]:
        """Process market data and generate signals"""
        if not self.enabled or data.symbol not in self.symbols:
            return None
        
        try:
            # Update price history
            self._update_price_history(data)
            
            # Generate signal using strategy logic
            signal = await self._generate_signal(data)
            
            if signal and signal.confidence >= self.min_confidence:
                # Avoid duplicate signals
                if self._should_generate_signal(signal):
                    self.signals_generated += 1
                    self.last_signals[data.symbol] = signal
                    
                    # Publish signal
                    await self.event_bus.publish(Event(
                        event_type=EventType.STRATEGY_SIGNAL,
                        data=signal,
                        timestamp=datetime.now(),
                        source=self.strategy_id
                    ))
                    
                    logger.info(f"Strategy {self.strategy_id} generated signal: {signal}")
                    return signal
            
        except Exception as e:
            logger.error(f"Error in strategy {self.strategy_id} processing data: {e}")
        
        return None
    
    def _update_price_history(self, data: MarketData):
        """Update price history for symbol"""
        if data.symbol not in self.price_history:
            self.price_history[data.symbol] = deque(maxlen=self.lookback_period * 2)
        
        self.price_history[data.symbol].append({
            'price': data.price,
            'timestamp': data.timestamp,
            'volume': data.volume
        })
    
    def _should_generate_signal(self, signal: Signal) -> bool:
        """Check if signal should be generated (avoid duplicates)"""
        last_signal = self.last_signals.get(signal.symbol)
        
        if not last_signal:
            return True
        
        # Don't generate same signal within short time window
        time_diff = (signal.timestamp - last_signal.timestamp).total_seconds()
        if time_diff < 300:  # 5 minutes
            if last_signal.action == signal.action:
                return False
        
        return True
    
    def get_price_array(self, symbol: str, periods: int = None) -> np.ndarray:
        """Get price array for calculations"""
        if symbol not in self.price_history:
            return np.array([])
        
        prices = [p['price'] for p in self.price_history[symbol]]
        if periods:
            prices = prices[-periods:]
        
        return np.array(prices)
    
    @abstractmethod
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Strategy-specific signal generation logic"""
        pass
    
    async def cleanup(self):
        """Cleanup strategy resources"""
        logger.info(f"Strategy {self.strategy_id} cleanup complete. Generated {self.signals_generated} signals")