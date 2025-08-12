# data/storage.py - Fixed version
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from datetime import datetime
from typing import List
from core.interfaces import MarketData
from loguru import logger

Base = declarative_base()

class MarketDataRecord(Base):
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    volume = Column(Integer)
    bid = Column(Float)
    ask = Column(Float)
    
    # Optimize for time-series queries
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_timestamp', 'timestamp'),
    )

class DatabaseManager:
    """Async database manager for persistent storage"""
    
    def __init__(self, database_url: str):
        # Fix the database URL for async operations
        if database_url.startswith('sqlite:///'):
            # Convert to async SQLite URL
            try:
                import aiosqlite
                database_url = database_url.replace('sqlite:///', 'sqlite+aiosqlite:///')
            except ImportError:
                logger.warning("aiosqlite not installed, database will be disabled")
                raise ImportError("aiosqlite required for async SQLite operations")
        
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
        
    async def initialize(self):
        """Create engine and tables"""
        try:
            self.engine = create_async_engine(self.database_url, echo=False)
            self.session_factory = sessionmaker(
                self.engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def store_market_data(self, data: List[MarketData]):
        """Store market data records"""
        if not self.session_factory:
            logger.warning("Database not initialized, skipping data storage")
            return
            
        try:
            async with self.session_factory() as session:
                records = [
                    MarketDataRecord(
                        symbol=d.symbol,
                        price=d.price,
                        timestamp=d.timestamp,
                        volume=d.volume,
                        bid=d.bid,
                        ask=d.ask
                    )
                    for d in data
                ]
                
                session.add_all(records)
                await session.commit()
                logger.debug(f"Stored {len(records)} market data records")
                
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
    
    async def cleanup(self):
        """Cleanup database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Alternative: Simple dummy database manager
class DummyDatabaseManager:
    """Dummy database manager when database is not available"""
    
    def __init__(self, database_url: str = None):
        logger.info("Using dummy database manager - no persistent storage")
    
    async def initialize(self):
        pass
    
    async def store_market_data(self, data: List[MarketData]):
        pass
    
    async def cleanup(self):
        pass