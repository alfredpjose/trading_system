# data/providers.py - Enhanced version with better logging

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncGenerator
import json
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from loguru import logger

from core.interfaces import IDataProvider, MarketData
from core.exceptions import ConnectionError
from .cache import DataCache

class IBKRDataProvider(EWrapper, EClient, IDataProvider):
    """IBKR data provider with connection management"""
    
    def __init__(self, host: str, port: int, client_id: int, cache):
        EClient.__init__(self, self)
        self.host = host
        self.port = port
        self.client_id = client_id
        self.cache = cache
        
        self._connected = False
        self._data_queue = asyncio.Queue()
        self._req_id_counter = 1000
        self._subscriptions: Dict[int, str] = {}
        self._connection_lock = asyncio.Lock()
        
        # Add counters for debugging
        self._price_updates_received = 0
        self._last_price_time = None
        
    async def connect(self) -> bool:
        """Connect to IBKR with retry logic"""
        async with self._connection_lock:
            if self._connected:
                return True
                
            try:
                # Connect in a thread to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    super().connect, 
                    self.host, 
                    self.port, 
                    self.client_id
                )
                
                # Start the message processing thread
                self._thread = asyncio.create_task(self._run_message_loop())
                
                # Wait for connection confirmation
                await asyncio.sleep(2)
                
                if self._connected:
                    logger.info(f"Connected to IBKR at {self.host}:{self.port}")
                    return True
                else:
                    raise ConnectionError("Failed to establish IBKR connection")
                    
            except Exception as e:
                logger.error(f"IBKR connection failed: {e}")
                raise ConnectionError(f"IBKR connection failed: {e}")
    
    async def _run_message_loop(self):
        """Run IBKR message processing loop"""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.run)
        except Exception as e:
            logger.error(f"IBKR message loop error: {e}")
            self._connected = False
    
    def connectAck(self):
        """Connection acknowledgment from IBKR"""
        self._connected = True
        logger.debug("IBKR connection acknowledged")
    
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        """Handle IBKR errors"""
        if errorCode in [2104, 2106, 2158]:  # Info messages
            logger.debug(f"IBKR Info {errorCode}: {errorString}")
        elif errorCode == 10089:  # Market data subscription warning
            logger.warning(f"IBKR Error {errorCode}: {errorString}")
        else:
            logger.warning(f"IBKR Error {errorCode}: {errorString}")
    
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
        """Handle real-time price updates"""
        if reqId in self._subscriptions:
            symbol = self._subscriptions[reqId]
            
            # Log different tick types for debugging
            tick_names = {1: "bid", 2: "ask", 4: "last", 6: "high", 7: "low", 9: "close"}
            tick_name = tick_names.get(tickType, f"type_{tickType}")
            
            # Only process last price for now, but log all types
            if tickType in [1, 2, 4]:  # bid, ask, last
                logger.debug(f"Price update: {symbol} {tick_name} = ${price:.2f}")
                self._price_updates_received += 1
                self._last_price_time = datetime.now()
            
            if tickType == 4:  # Last price - this is what we want for strategies
                market_data = MarketData(
                    symbol=symbol,
                    price=price,
                    timestamp=datetime.now(),
                    volume=None,
                    bid=None,
                    ask=None
                )
                
                # Put data in queue (non-blocking)
                try:
                    self._data_queue.put_nowait(market_data)
                    logger.info(f"📈 Market data: {symbol} = ${price:.2f}")
                    
                    # Cache the data if cache is available
                    if hasattr(self.cache, 'set_market_data'):
                        asyncio.create_task(self.cache.set_market_data(symbol, market_data))
                        
                except asyncio.QueueFull:
                    logger.warning(f"Data queue full, dropping {symbol} price update")
    
    def tickSize(self, reqId: int, tickType: int, size: int):
        """Handle volume updates"""
        if reqId in self._subscriptions:
            symbol = self._subscriptions[reqId]
            if tickType == 5:  # Last size (volume)
                logger.debug(f"Volume update: {symbol} volume = {size}")
    
    def tickString(self, reqId: int, tickType: int, value: str):
        """Handle string tick data"""
        if reqId in self._subscriptions:
            symbol = self._subscriptions[reqId]
            logger.debug(f"String data: {symbol} type {tickType} = {value}")
    
    async def subscribe(self, symbols: List[str]) -> AsyncGenerator[MarketData, None]:
        """Subscribe to real-time market data"""
        if not self._connected:
            await self.connect()
        
        # Subscribe to each symbol
        for symbol in symbols:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            req_id = self._req_id_counter
            self._req_id_counter += 1
            self._subscriptions[req_id] = symbol
            
            # Subscribe in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.reqMktData,
                req_id, contract, "", False, False, []
            )
            
            logger.info(f"Subscribed to {symbol} (req_id: {req_id})")
            await asyncio.sleep(0.1)  # Rate limit subscriptions
        
        # Start periodic status logging
        asyncio.create_task(self._log_data_status())
        
        # Yield data as it arrives
        while True:
            try:
                data = await asyncio.wait_for(self._data_queue.get(), timeout=5.0)
                yield data
            except asyncio.TimeoutError:
                # Log status every 5 seconds when no data
                logger.debug(f"No market data received in last 5 seconds. Total updates: {self._price_updates_received}")
                continue
            except Exception as e:
                logger.error(f"Error in data subscription: {e}")
                break
    
    async def _log_data_status(self):
        """Periodically log data reception status"""
        while self._connected:
            await asyncio.sleep(30)  # Every 30 seconds
            logger.info(f"📊 Data Status: {self._price_updates_received} price updates received")
            if self._last_price_time:
                time_since = (datetime.now() - self._last_price_time).total_seconds()
                logger.info(f"⏰ Last price update: {time_since:.0f} seconds ago")
            else:
                logger.info("⏰ No price updates received yet")
    
    async def get_historical(self, symbol: str, days: int) -> List[MarketData]:
        """Get historical data with caching"""
        # Check cache first
        if hasattr(self.cache, 'get_historical_data'):
            cached_data = await self.cache.get_historical_data(symbol, days)
            if cached_data:
                logger.debug(f"Retrieved {len(cached_data)} historical points for {symbol} from cache")
                return cached_data
        
        # Fetch from IBKR if not cached
        # Implementation would use reqHistoricalData
        # For now, return empty list (would implement full historical data fetching)
        logger.warning(f"Historical data for {symbol} not implemented yet")
        return []
    
    async def disconnect(self):
    """Disconnect from IBKR"""
    if self._connected:
        try:
            # Properly await the disconnect to avoid RuntimeWarning
            self._connected = False
            
            # Cancel the message loop task first
            if hasattr(self, '_thread') and not self._thread.done():
                self._thread.cancel()
                try:
                    await self._thread
                except asyncio.CancelledError:
                    pass  # Expected when cancelling
            
            # Use executor for the actual disconnect to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, super().disconnect
            )
            
            logger.info("Disconnected from IBKR")
            
        except Exception as e:
            logger.error(f"Error during IBKR disconnect: {e}")
    else:
        logger.debug("IBKR was not connected")
                
    def get_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            "connected": self._connected,
            "subscriptions": len(self._subscriptions),
            "price_updates_received": self._price_updates_received,
            "last_price_time": self._last_price_time.isoformat() if self._last_price_time else None
        }