# monitoring/alerts.py
from dataclasses import dataclass

from enum import Enum
from typing import Dict, List, Callable
import asyncio
from datetime import datetime
from loguru import logger

from monitoring.metrics import SystemMetrics
class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    level: AlertLevel
    message: str
    timestamp: datetime
    component: str
    metadata: Dict = None

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = {
            AlertLevel.INFO: [],
            AlertLevel.WARNING: [],
            AlertLevel.CRITICAL: []
        }
        self.recent_alerts: List[Alert] = []
        self.max_alerts = 1000
        
        # Alert thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'daily_loss_pct': 0.05,  # 5%
            'order_latency': 5000.0,  # 5 seconds
            'data_latency': 1000.0    # 1 second
        }
    
    def add_handler(self, level: AlertLevel, handler: Callable):
        """Add alert handler function"""
        self.alert_handlers[level].append(handler)
    
    async def check_system_health(self, metrics: SystemMetrics, risk_summary: Dict):
        """Check system health and generate alerts"""
        alerts = []
        
        # CPU usage alert
        if metrics.cpu_usage > self.thresholds['cpu_usage']:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                timestamp=datetime.now(),
                component="system",
                metadata={'cpu_usage': metrics.cpu_usage}
            ))
        
        # Memory usage alert
        if metrics.memory_usage > self.thresholds['memory_usage']:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High memory usage: {metrics.memory_usage:.1f}%",
                timestamp=datetime.now(),
                component="system",
                metadata={'memory_usage': metrics.memory_usage}
            ))
        
        # Trading performance alerts
        daily_pnl_pct = risk_summary.get('daily_pnl_pct', 0.0)
        if daily_pnl_pct < -self.thresholds['daily_loss_pct']:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Daily loss limit approached: {daily_pnl_pct:.2%}",
                timestamp=datetime.now(),
                component="risk",
                metadata={'daily_pnl_pct': daily_pnl_pct}
            ))
        
        # Circuit breaker alert
        if risk_summary.get('circuit_breaker_active', False):
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message="Circuit breaker activated - trading halted",
                timestamp=datetime.now(),
                component="risk",
                metadata={'circuit_breaker': True}
            ))
        
        # Latency alerts
        if metrics.order_latency > self.thresholds['order_latency']:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High order latency: {metrics.order_latency:.0f}ms",
                timestamp=datetime.now(),
                component="execution",
                metadata={'order_latency': metrics.order_latency}
            ))
        
        # Process all alerts
        for alert in alerts:
            await self._process_alert(alert)
    
    async def _process_alert(self, alert: Alert):
        """Process and distribute alert"""
        # Add to recent alerts
        self.recent_alerts.append(alert)
        if len(self.recent_alerts) > self.max_alerts:
            self.recent_alerts = self.recent_alerts[-self.max_alerts:]
        
        # Log alert
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(f"ALERT: {alert.message}")
        elif alert.level == AlertLevel.WARNING:
            logger.warning(f"ALERT: {alert.message}")
        else:
            logger.info(f"ALERT: {alert.message}")
        
        # Call handlers
        handlers = self.alert_handlers.get(alert.level, [])
        if handlers:
            await asyncio.gather(
                *[handler(alert) for handler in handlers],
                return_exceptions=True
            )
    
    def get_recent_alerts(self, level: AlertLevel = None) -> List[Alert]:
        """Get recent alerts, optionally filtered by level"""
        if level:
            return [a for a in self.recent_alerts if a.level == level]
        return self.recent_alerts.copy()