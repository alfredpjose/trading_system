# data/storage.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from datetime import datetime
from typing import List

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
        self.engine = create_async_engine(database_url, echo=False)
        self.session_factory = sessionmaker(
            self.engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )
    
    async def initialize(self):
        """Create tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized")
    
    async def store_market_data(self, data: List[MarketData]):
        """Store market data records"""
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
    
    async def cleanup(self):
        """Cleanup database connections"""
        await self.engine.dispose()