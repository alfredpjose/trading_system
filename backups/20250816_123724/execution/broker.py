# execution/broker.py
import asyncio
from datetime import datetime
from typing import Dict, AsyncGenerator, Optional
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order as IBOrder
from loguru import logger

from core.interfaces import IBroker, Order, Fill, OrderAction, OrderType
from core.exceptions import ConnectionError, OrderError

class IBKRBroker(EWrapper, EClient, IBroker):
    """IBKR broker implementation"""
    
    def __init__(self, host: str, port: int, client_id: int):
        EClient.__init__(self, self)
        self.host = host
        self.port = port
        self.client_id = client_id
        
        self._connected = False
        self._next_order_id = 1
        self._fills_queue = asyncio.Queue()
        self._positions: Dict[str, int] = {}
        self._connection_lock = asyncio.Lock()
    
    async def connect(self) -> bool:
        """Connect to IBKR"""
        async with self._connection_lock:
            if self._connected:
                return True
            
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, super().connect, self.host, self.port, self.client_id
                )
                
                # Start message loop
                self._thread = asyncio.create_task(self._run_message_loop())
                await asyncio.sleep(2)
                
                if self._connected:
                    logger.info(f"Connected to IBKR broker at {self.host}:{self.port}")
                    return True
                else:
                    raise ConnectionError("Failed to establish broker connection")
                    
            except Exception as e:
                logger.error(f"Broker connection failed: {e}")
                raise ConnectionError(f"Broker connection failed: {e}")
    
    async def _run_message_loop(self):
        """Run IBKR message processing loop"""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.run)
        except Exception as e:
            logger.error(f"Broker message loop error: {e}")
            self._connected = False
    
    def connectAck(self):
        """Connection acknowledgment"""
        self._connected = True
    
    def nextValidId(self, orderId: int):
        """Set next valid order ID"""
        self._next_order_id = orderId
        logger.debug(f"Next valid order ID: {orderId}")
    
    def execDetails(self, reqId: int, contract, execution):
        """Handle execution details"""
        fill = Fill(
            order_id=str(execution.orderId),
            symbol=contract.symbol,
            action=OrderAction.BUY if execution.side == "BOT" else OrderAction.SELL,
            quantity=int(execution.shares),
            price=execution.price,
            timestamp=datetime.now(),
            commission=0.0  # Will be updated in commissionReport
        )
        
        try:
            self._fills_queue.put_nowait(fill)
        except asyncio.QueueFull:
            logger.warning("Fills queue full, dropping fill notification")
    
    def position(self, account: str, contract, position: float, avgCost: float):
        """Handle position updates"""
        if contract.secType == "STK":
            self._positions[contract.symbol] = int(position)
    
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        """Handle errors"""
        if errorCode in [2104, 2106, 2158]:
            logger.debug(f"IBKR Info {errorCode}: {errorString}")
        else:
            logger.warning(f"IBKR Error {errorCode}: {errorString}")
    
    async def place_order(self, order: Order) -> str:
        """Place order with IBKR"""
        if not self._connected:
            raise OrderError("Not connected to broker")
        
        try:
            # Create IBKR contract
            contract = Contract()
            contract.symbol = order.symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Create IBKR order
            ib_order = IBOrder()
            ib_order.action = order.action.value
            ib_order.orderType = order.order_type.value
            ib_order.totalQuantity = order.quantity
            
            if order.price and order.order_type == OrderType.LIMIT:
                ib_order.lmtPrice = order.price
            
            # Place order
            order_id = self._next_order_id
            self._next_order_id += 1
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.placeOrder, order_id, contract, ib_order
            )
            
            logger.info(f"Placed order {order_id}: {order}")
            return str(order_id)
            
        except Exception as e:
            logger.error(f"Failed to place order {order}: {e}")
            raise OrderError(f"Order placement failed: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.cancelOrder, int(order_id)
            )
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_positions(self) -> Dict[str, int]:
        """Get current positions"""
        # Request position updates
        await asyncio.get_event_loop().run_in_executor(
            None, self.reqPositions
        )
        await asyncio.sleep(0.5)  # Wait for position updates
        return self._positions.copy()
    
    async def get_fills(self) -> AsyncGenerator[Fill, None]:
        """Get order fills"""
        while True:
            try:
                fill = await self._fills_queue.get()
                yield fill
            except Exception as e:
                logger.error(f"Error getting fills: {e}")
                break