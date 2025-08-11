# execution/engine.py
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from core.interfaces import (
    IEventBus, IBroker, IRiskManager, 
    Event, EventType, Order, Fill, Signal
)
from core.exceptions import OrderError, RiskError

class ExecutionEngine:
    """Main execution engine coordinating orders and risk"""
    
    def __init__(self, broker: IBroker, risk_manager: IRiskManager, event_bus: IEventBus):
        self.broker = broker
        self.risk_manager = risk_manager
        self.event_bus = event_bus
        self._running = False
        self._pending_orders: Dict[str, Order] = {}
        
    async def start(self):
        """Start the execution engine"""
        self._running = True
        
        # Connect to broker
        await self.broker.connect()
        
        # Start listening for signals
        asyncio.create_task(self._process_signals())
        # Start listening for fills
        asyncio.create_task(self._process_fills())
        
        logger.info("Execution engine started")
    
    async def stop(self):
        """Stop the execution engine"""
        self._running = False
        logger.info("Execution engine stopped")
    
    async def _process_signals(self):
        """Process trading signals from strategies"""
        async for event in self.event_bus.subscribe(EventType.STRATEGY_SIGNAL):
            if not self._running:
                break
                
            try:
                signal: Signal = event.data
                await self._handle_signal(signal)
            except Exception as e:
                logger.error(f"Error processing signal: {e}")
    
    async def _process_fills(self):
        """Process order fills from broker"""
        async for fill in self.broker.get_fills():
            if not self._running:
                break
                
            try:
                await self._handle_fill(fill)
            except Exception as e:
                logger.error(f"Error processing fill: {e}")
    
    async def _handle_signal(self, signal: Signal):
        """Convert signal to order and execute"""
        try:
            # Get current portfolio value for risk checks
            positions = await self.broker.get_positions()
            portfolio_value = await self._calculate_portfolio_value(positions)
            
            # Create order from signal
            order = Order(
                symbol=signal.symbol,
                action=signal.action,
                quantity=self._calculate_position_size(signal, portfolio_value),
                order_type=OrderType.MARKET,
                strategy_id=signal.strategy_id
            )
            
            # Risk validation
            if not await self.risk_manager.validate_order(order, portfolio_value):
                logger.warning(f"Order rejected by risk manager: {order}")
                return
            
            # Execute order
            order_id = await self.broker.place_order(order)
            order.order_id = order_id
            self._pending_orders[order_id] = order
            
            logger.info(f"Order placed: {order}")
            
        except Exception as e:
            logger.error(f"Failed to handle signal {signal}: {e}")
            raise OrderError(f"Signal processing failed: {e}")
    
    async def _handle_fill(self, fill: Fill):
        """Process order fill"""
        # Remove from pending orders
        if fill.order_id in self._pending_orders:
            del self._pending_orders[fill.order_id]
        
        # Update risk metrics
        await self.risk_manager.update_metrics(fill)
        
        # Publish fill event
        await self.event_bus.publish(Event(
            event_type=EventType.ORDER_FILL,
            data=fill,
            timestamp=datetime.now(),
            source="execution_engine"
        ))
        
        logger.info(f"Order filled: {fill}")
    
    def _calculate_position_size(self, signal: Signal, portfolio_value: float) -> int:
        """Calculate position size based on signal confidence and portfolio value"""
        # Simple position sizing: use confidence as a multiplier
        base_size = 100  # Base position size
        confidence_multiplier = min(signal.confidence, 2.0)  # Cap at 2x
        
        position_size = int(base_size * confidence_multiplier)
        
        # Ensure position doesn't exceed portfolio limits
        max_position_value = portfolio_value * 0.1  # 10% max position
        if hasattr(signal, 'metadata') and signal.metadata:
            estimated_price = signal.metadata.get('price', 100)  # Default price
            max_shares = int(max_position_value / estimated_price)
            position_size = min(position_size, max_shares)
        
        return max(position_size, 1)  # Minimum 1 share
    
    async def _calculate_portfolio_value(self, positions: Dict[str, int]) -> float:
        """Calculate current portfolio value"""
        # Simplified calculation - in production would use real-time prices
        return 100000.0  # Placeholder
