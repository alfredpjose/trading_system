# monitoring/metrics.py
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from dataclasses import dataclass, asdict
from loguru import logger

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_strategies: int
    pending_orders: int
    fills_today: int
    daily_pnl: float
    portfolio_value: float
    data_feed_latency: float
    order_latency: float
    
class MetricsCollector:
    """Collects and aggregates system metrics"""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 1440  # 24 hours of 1-minute data
        self._running = False
        
        # Components to monitor
        self.execution_engine = None
        self.data_provider = None
        self.risk_manager = None
        
    def set_components(self, execution_engine, data_provider, risk_manager):
        """Set system components to monitor"""
        self.execution_engine = execution_engine
        self.data_provider = data_provider
        self.risk_manager = risk_manager
    
    async def start_collection(self):
        """Start metrics collection"""
        self._running = True
        asyncio.create_task(self._collect_metrics_loop())
        logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self._running = False
        logger.info("Metrics collection stopped")
    
    async def _collect_metrics_loop(self):
        """Main metrics collection loop"""
        while self._running:
            try:
                metrics = await self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history:]
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_current_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        import psutil
        
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # Trading metrics
        active_strategies = 0
        pending_orders = 0
        fills_today = 0
        daily_pnl = 0.0
        portfolio_value = 100000.0  # Default
        
        if self.execution_engine:
            pending_orders = len(getattr(self.execution_engine, '_pending_orders', {}))
        
        if self.risk_manager:
            risk_summary = getattr(self.risk_manager, 'get_risk_summary', lambda: {})()
            daily_pnl = risk_summary.get('daily_pnl', 0.0)
            portfolio_value = risk_summary.get('current_capital', 100000.0)
            fills_today = risk_summary.get('daily_trades', 0)
        
        # Performance metrics (simplified)
        data_feed_latency = 50.0  # ms
        order_latency = 100.0     # ms
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_strategies=active_strategies,
            pending_orders=pending_orders,
            fills_today=fills_today,
            daily_pnl=daily_pnl,
            portfolio_value=portfolio_value,
            data_feed_latency=data_feed_latency,
            order_latency=order_latency
        )
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get latest metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of recent metrics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-60:]  # Last hour
        
        return {
            'avg_cpu': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            'avg_memory': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            'total_fills': sum(m.fills_today for m in recent_metrics),
            'current_pnl': recent_metrics[-1].daily_pnl,
            'avg_data_latency': sum(m.data_feed_latency for m in recent_metrics) / len(recent_metrics),
            'avg_order_latency': sum(m.order_latency for m in recent_metrics) / len(recent_metrics)
        }
