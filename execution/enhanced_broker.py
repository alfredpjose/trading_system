# execution/enhanced_broker.py
from typing import Dict, List, Optional, AsyncGenerator
import asyncio
from datetime import datetime
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order as IBOrder
from loguru import logger

from execution.broker import IBKRBroker
from execution.contract_builder import UniversalContractBuilder

class EnhancedIBKRBroker(IBKRBroker):
    """Enhanced IBKR broker supporting all asset classes"""
    
    def __init__(self, host: str, port: int, client_id: int):
        super().__init__(host, port, client_id)
        self.contract_builder = UniversalContractBuilder()
        self.multi_market_positions = {}
        self.account_summary = {}
        
    async def get_multi_market_positions(self) -> Dict[str, List]:
        """Get positions across all asset classes"""
        await self.reqPositions()
        await asyncio.sleep(2)
        
        positions_by_market = {
            'stocks': [],
            'forex': [],
            'futures': [],
            'commodities': [],
            'options': [],
            'crypto': []
        }
        
        for position in self._positions.values():
            asset_class = self._determine_asset_class(position)
            positions_by_market[asset_class].append(position)
        
        return positions_by_market
    
    def _determine_asset_class(self, position) -> str:
        """Determine asset class from position"""
        contract = position.contract
        
        if contract.secType == 'STK':
            return 'stocks'
        elif contract.secType == 'CASH':
            return 'forex'
        elif contract.secType == 'FUT':
            if contract.symbol in ['CL', 'NG', 'HG', 'GC', 'SI']:
                return 'commodities'
            return 'futures'
        elif contract.secType == 'OPT':
            return 'options'
        elif contract.secType == 'CRYPTO':
            return 'crypto'
        else:
            return 'other'
    
    async def place_multi_market_order(self, symbol: str, asset_class: str, 
                                      action: str, quantity: int, **kwargs) -> str:
        """Place order for any asset class"""
        try:
            # Build appropriate contract
            contract = self.contract_builder.build_contract(
                symbol, asset_class, **kwargs
            )
            
            # Create order
            order = IBOrder()
            order.action = action
            order.orderType = kwargs.get('order_type', 'MKT')
            order.totalQuantity = quantity
            
            if 'price' in kwargs and order.orderType == 'LMT':
                order.lmtPrice = kwargs['price']
            
            # Place order
            order_id = self._next_order_id
            self._next_order_id += 1
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.placeOrder, order_id, contract, order
            )
            
            logger.info(f"Placed {asset_class} order: {action} {quantity} {symbol}")
            return str(order_id)
            
        except Exception as e:
            logger.error(f"Failed to place {asset_class} order for {symbol}: {e}")
            raise
