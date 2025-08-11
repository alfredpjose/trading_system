# data/cache.py
import asyncio
import json
import redis.asyncio as redis
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from loguru import logger

from core.interfaces import MarketData

class DataCache:
    """Redis-based data cache for market data"""
    
    def __init__(self, redis_url: str, ttl: int = 60):
        self.redis_url = redis_url
        self.ttl = ttl
        self._redis: Optional[redis.Redis] = None
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self._redis = None
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.close()
            logger.info("Disconnected from Redis cache")
    
    async def set_market_data(self, symbol: str, data: MarketData):
        """Cache market data point"""
        if not self._redis:
            return
            
        try:
            key = f"market_data:{symbol}"
            value = {
                "price": data.price,
                "timestamp": data.timestamp.isoformat(),
                "volume": data.volume,
                "bid": data.bid,
                "ask": data.ask
            }
            
            await self._redis.setex(key, self.ttl, json.dumps(value))
            
            # Also add to time series for historical data
            ts_key = f"history:{symbol}"
            await self._redis.zadd(
                ts_key, 
                {json.dumps(value): data.timestamp.timestamp()}
            )
            
            # Cleanup old historical data (keep last 1000 points)
            await self._redis.zremrangebyrank(ts_key, 0, -1001)
            
        except Exception as e:
            logger.error(f"Failed to cache market data for {symbol}: {e}")
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data from cache"""
        if not self._redis:
            return None
            
        try:
            key = f"market_data:{symbol}"
            value = await self._redis.get(key)
            
            if value:
                data = json.loads(value)
                return MarketData(
                    symbol=symbol,
                    price=data["price"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    volume=data.get("volume"),
                    bid=data.get("bid"),
                    ask=data.get("ask")
                )
        except Exception as e:
            logger.error(f"Failed to retrieve cached data for {symbol}: {e}")
        
        return None
    
    async def get_historical_data(self, symbol: str, days: int) -> List[MarketData]:
        """Get historical data from cache"""
        if not self._redis:
            return []
            
        try:
            ts_key = f"history:{symbol}"
            cutoff_time = (datetime.now() - timedelta(days=days)).timestamp()
            
            # Get data points from the time series
            data_points = await self._redis.zrangebyscore(
                ts_key, 
                cutoff_time, 
                datetime.now().timestamp(),
                withscores=True
            )
            
            historical_data = []
            for data_json, timestamp in data_points:
                data = json.loads(data_json)
                historical_data.append(MarketData(
                    symbol=symbol,
                    price=data["price"],
                    timestamp=datetime.fromtimestamp(timestamp),
                    volume=data.get("volume"),
                    bid=data.get("bid"),
                    ask=data.get("ask")
                ))
            
            return sorted(historical_data, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error(f"Failed to retrieve historical data for {symbol}: {e}")
            return []
    
    async def clear_symbol_data(self, symbol: str):
        """Clear all cached data for a symbol"""
        if not self._redis:
            return
            
        try:
            await self._redis.delete(f"market_data:{symbol}")
            await self._redis.delete(f"history:{symbol}")
            logger.info(f"Cleared cached data for {symbol}")
        except Exception as e:
            logger.error(f"Failed to clear data for {symbol}: {e}")