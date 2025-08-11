# core/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio

class OrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"

class OrderAction(Enum):
    BUY = "BUY"
    SELL = "SELL"

class EventType(Enum):
    MARKET_DATA = "market_data"
    ORDER_FILL = "order_fill"
    STRATEGY_SIGNAL = "strategy_signal"
    SYSTEM_ERROR = "system_error"

@dataclass
class MarketData:
    symbol: str
    price: float
    timestamp: datetime
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

@dataclass
class Order:
    symbol: str
    action: OrderAction
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    order_id: Optional[str] = None
    strategy_id: str = ""

@dataclass
class Fill:
    order_id: str
    symbol: str
    action: OrderAction
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0

@dataclass
class Signal:
    symbol: str
    action: OrderAction
    confidence: float
    strategy_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class Event:
    event_type: EventType
    data: Any
    timestamp: datetime
    source: str = ""

# Interfaces
class IDataProvider(ABC):
    """Interface for market data providers"""
    
    @abstractmethod
    async def connect(self) -> bool:
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> AsyncGenerator[MarketData, None]:
        pass
    
    @abstractmethod
    async def get_historical(self, symbol: str, days: int) -> List[MarketData]:
        pass
    
    @abstractmethod
    async def disconnect(self):
        pass

class IBroker(ABC):
    """Interface for order execution"""
    
    @abstractmethod
    async def connect(self) -> bool:
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, int]:
        pass
    
    @abstractmethod
    async def get_fills(self) -> AsyncGenerator[Fill, None]:
        pass

class IStrategy(ABC):
    """Interface for trading strategies"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]):
        pass
    
    @abstractmethod
    async def process_data(self, data: MarketData) -> Optional[Signal]:
        pass
    
    @abstractmethod
    async def cleanup(self):
        pass
    
    @property
    @abstractmethod
    def strategy_id(self) -> str:
        pass

class IEventBus(ABC):
    """Interface for event handling"""
    
    @abstractmethod
    async def publish(self, event: Event):
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: EventType) -> AsyncGenerator[Event, None]:
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: EventType):
        pass

class IRiskManager(ABC):
    """Interface for risk management"""
    
    @abstractmethod
    async def validate_order(self, order: Order, portfolio_value: float) -> bool:
        pass
    
    @abstractmethod
    async def check_position_limits(self, symbol: str, quantity: int) -> bool:
        pass
    
    @abstractmethod
    async def update_metrics(self, fill: Fill):
        pass
