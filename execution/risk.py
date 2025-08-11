# execution/risk.py
from datetime import datetime, timedelta
from typing import Dict, List
from loguru import logger

from core.interfaces import IRiskManager, Order, Fill
from core.exceptions import RiskError
from config.settings import RiskManagementConfig as RiskConfig

class RiskManager(IRiskManager):
    """Risk management implementation"""
    
    def __init__(self, config: RiskConfig, initial_capital: float):
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.peak_capital = initial_capital
        self.positions: Dict[str, int] = {}
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
        
        # Circuit breaker
        self.circuit_breaker_active = False
        self.consecutive_losses = 0
        
    async def validate_order(self, order: Order, portfolio_value: float) -> bool:
        """Validate order against risk parameters"""
        try:
            # Update daily stats if needed
            await self._check_daily_reset()
            
            # Circuit breaker check
            if self.circuit_breaker_active:
                logger.warning("Circuit breaker active, rejecting order")
                return False
            
            # Portfolio value check
            if portfolio_value <= 0:
                logger.error("Invalid portfolio value")
                return False
            
            # Daily loss limit
            daily_pnl_pct = self.daily_pnl / self.current_capital
            if daily_pnl_pct <= -self.config.daily_loss_limit:
                logger.warning(f"Daily loss limit exceeded: {daily_pnl_pct:.2%}")
                return False
            
            # Max drawdown check
            drawdown = (self.peak_capital - portfolio_value) / self.peak_capital
            if drawdown >= self.config.max_drawdown:
                logger.warning(f"Max drawdown exceeded: {drawdown:.2%}")
                self._activate_circuit_breaker()
                return False
            
            # Position size check
            estimated_order_value = order.quantity * 100  # Simplified price estimation
            position_pct = estimated_order_value / portfolio_value
            
            if position_pct > self.config.max_position_size:
                logger.warning(f"Position size too large: {position_pct:.2%}")
                return False
            
            # Max positions check
            if len(self.positions) >= self.config.max_open_positions:
                if order.symbol not in self.positions:
                    logger.warning("Maximum open positions reached")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Risk validation error: {e}")
            return False
    
    async def check_position_limits(self, symbol: str, quantity: int) -> bool:
        """Check if position change is within limits"""
        current_position = self.positions.get(symbol, 0)
        new_position = current_position + quantity
        
        # Check if position would exceed limits
        max_position = int(self.current_capital * self.config.max_position_size / 100)
        
        return abs(new_position) <= max_position
    
    async def update_metrics(self, fill: Fill):
        """Update risk metrics after fill"""
        try:
            # Update positions
            current_pos = self.positions.get(fill.symbol, 0)
            
            if fill.action.value == "BUY":
                self.positions[fill.symbol] = current_pos + fill.quantity
            else:
                self.positions[fill.symbol] = current_pos - fill.quantity
            
            # Remove zero positions
            if self.positions.get(fill.symbol, 0) == 0:
                self.positions.pop(fill.symbol, None)
            
            # Update PnL (simplified)
            trade_value = fill.quantity * fill.price
            if fill.action.value == "SELL":
                self.daily_pnl += trade_value - fill.commission
            else:
                self.daily_pnl -= trade_value + fill.commission
            
            # Update peak capital
            current_value = self.current_capital + self.daily_pnl
            if current_value > self.peak_capital:
                self.peak_capital = current_value
                self.consecutive_losses = 0  # Reset on new peak
            
            self.daily_trades += 1
            
            logger.debug(f"Risk metrics updated: Daily PnL: ${self.daily_pnl:.2f}, "
                        f"Positions: {len(self.positions)}")
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    async def _check_daily_reset(self):
        """Check if daily stats need reset"""
        today = datetime.now().date()
        if today > self.last_reset:
            # Reset daily stats
            self.current_capital += self.daily_pnl
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = today
            
            # Deactivate circuit breaker on new day
            if self.circuit_breaker_active:
                self.circuit_breaker_active = False
                logger.info("Circuit breaker deactivated for new trading day")
            
            logger.info(f"Daily stats reset. Capital: ${self.current_capital:.2f}")
    
    def _activate_circuit_breaker(self):
        """Activate circuit breaker to stop trading"""
        self.circuit_breaker_active = True
        logger.critical("CIRCUIT BREAKER ACTIVATED - Trading halted due to risk limits")
    
    def get_risk_summary(self) -> Dict:
        """Get current risk summary"""
        current_value = self.current_capital + self.daily_pnl
        drawdown = (self.peak_capital - current_value) / self.peak_capital
        daily_pnl_pct = self.daily_pnl / self.current_capital
        
        return {
            "current_capital": current_value,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "drawdown": drawdown,
            "positions_count": len(self.positions),
            "daily_trades": self.daily_trades,
            "circuit_breaker_active": self.circuit_breaker_active,
            "consecutive_losses": self.consecutive_losses
        }