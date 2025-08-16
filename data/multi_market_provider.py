# data/multi_market_provider.py
from typing import List, Dict, AsyncGenerator, Optional
import asyncio
from datetime import datetime
from loguru import logger

from data.providers import IBKRDataProvider
from execution.contract_builder import UniversalContractBuilder
from core.interfaces import MarketData

class MultiMarketDataProvider(IBKRDataProvider):
    """Enhanced data provider supporting all markets"""
    
    def __init__(self, host: str, port: int, client_id: int, cache):
        super().__init__(host, port, client_id, cache)
        self.contract_builder = UniversalContractBuilder()
        self.market_subscriptions = {}
        self.asset_class_stats = {}
        
    async def subscribe_all_markets(self, market_config: Dict) -> AsyncGenerator[MarketData, None]:
        """Subscribe to all configured markets"""
        
        total_subscriptions = 0
        
        # Subscribe to each market
        for asset_class, config in market_config.items():
            symbols = config.get('symbols', [])
            logger.info(f"Subscribing to {len(symbols)} {asset_class} symbols")
            
            for symbol in symbols:
                try:
                    await self._subscribe_market_symbol(symbol, asset_class, config)
                    total_subscriptions += 1
                    await asyncio.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Failed to subscribe to {asset_class} {symbol}: {e}")
        
        logger.info(f"Total subscriptions: {total_subscriptions}")
        
        # Start monitoring and yielding data
        async for data in self._yield_multi_market_data():
            yield data
    
    async def _subscribe_market_symbol(self, symbol: str, asset_class: str, config: Dict):
        """Subscribe to individual market symbol"""
        try:
            # Build contract
            contract_params = config.get('contract_params', {})
            contract = self.contract_builder.build_contract(
                symbol, asset_class, **contract_params
            )
            
            # Get request ID
            req_id = self._req_id_counter
            self._req_id_counter += 1
            
            # Store subscription info
            self._subscriptions[req_id] = symbol
            self.market_subscriptions[symbol] = {
                'req_id': req_id,
                'asset_class': asset_class,
                'contract': contract,
                'subscribed_at': datetime.now()
            }
            
            # Subscribe with appropriate tick types
            tick_list = self._get_tick_list_for_asset_class(asset_class)
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.reqMktData,
                req_id, contract, tick_list, False, False, []
            )
            
            logger.debug(f"Subscribed to {asset_class} {symbol} (req_id: {req_id})")
            
        except Exception as e:
            logger.error(f"Subscription failed for {asset_class} {symbol}: {e}")
            raise
    
    def _get_tick_list_for_asset_class(self, asset_class: str) -> str:
        """Get appropriate tick list for asset class"""
        tick_lists = {
            'stocks': "233,236,258,293",      # Mark price, shortable, fundamentals, trade volume
            'forex': "233,236,293",           # Mark price, shortable, trade volume
            'commodities': "233,236,293",     # Mark price, shortable, trade volume
            'futures': "233,236,293,375",     # Mark price, shortable, volume, RT volume
            'options': "100,101,104,105,106,293", # Bid/ask size, Greeks, volume
            'crypto': "233,236,293"           # Mark price, shortable, volume
        }
        return tick_lists.get(asset_class, "233,236")
    
    async def _yield_multi_market_data(self) -> AsyncGenerator[MarketData, None]:
        """Yield market data from all subscribed markets"""
        while True:
            try:
                # Wait for data with timeout
                data = await asyncio.wait_for(self._data_queue.get(), timeout=10.0)
                
                # Enhance data with asset class info
                if data.symbol in self.market_subscriptions:
                    sub_info = self.market_subscriptions[data.symbol]
                    data.asset_class = sub_info['asset_class']
                
                # Update statistics
                self._update_asset_class_stats(data)
                
                yield data
                
            except asyncio.TimeoutError:
                # Log status every 10 seconds when no data
                self._log_subscription_status()
                continue
            except Exception as e:
                logger.error(f"Error in multi-market data stream: {e}")
                await asyncio.sleep(1)
    
    def _update_asset_class_stats(self, data: MarketData):
        """Update statistics by asset class"""
        asset_class = getattr(data, 'asset_class', 'unknown')
        
        if asset_class not in self.asset_class_stats:
            self.asset_class_stats[asset_class] = {
                'updates_received': 0,
                'last_update': None,
                'symbols': set()
            }
        
        stats = self.asset_class_stats[asset_class]
        stats['updates_received'] += 1
        stats['last_update'] = datetime.now()
        stats['symbols'].add(data.symbol)
    
    def _log_subscription_status(self):
        """Log current subscription status"""
        total_subscriptions = len(self.market_subscriptions)
        logger.info(f"Multi-market status: {total_subscriptions} subscriptions active")
        
        for asset_class, stats in self.asset_class_stats.items():
            symbol_count = len(stats['symbols'])
            updates = stats['updates_received']
            last_update = stats['last_update']
            
            time_since = "Never"
            if last_update:
                seconds_ago = (datetime.now() - last_update).total_seconds()
                time_since = f"{seconds_ago:.0f}s ago"
            
            logger.info(f"  {asset_class}: {symbol_count} symbols, {updates} updates, last: {time_since}")
    
    def get_market_statistics(self) -> Dict:
        """Get comprehensive market statistics"""
        return {
            'total_subscriptions': len(self.market_subscriptions),
            'by_asset_class': dict(self.asset_class_stats),
            'connection_status': self._connected,
            'data_updates_total': self._price_updates_received
        }
