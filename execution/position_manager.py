# execution/position_manager.py
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
from loguru import logger

@dataclass
class RealPosition:
    """Real position from IBKR"""
    symbol: str
    asset_class: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    currency: str
    exchange: str
    contract_info: Dict

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    name: str
    signals_generated: int
    trades_executed: int
    total_pnl: float
    win_rate: float
    avg_trade_duration: float
    max_drawdown: float
    sharpe_ratio: float
    active: bool

class RealPositionManager:
    """Manage actual IBKR paper trading positions"""
    
    def __init__(self, broker=None):
        self.broker = broker
        self.positions: Dict[str, RealPosition] = {}
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.last_update = None
        self.trade_history = []
        
    async def update_positions(self):
        """Get latest positions from IBKR"""
        if not self.broker or not self.broker.is_connected():
            logger.warning("Broker not connected - cannot update positions")
            return
        
        try:
            # Request account updates
            await self.broker.reqAccountUpdates(True, "")
            await asyncio.sleep(2)
            
            # Get positions
            raw_positions = await self.broker.get_positions()
            self.positions = self._convert_positions(raw_positions)
            self.last_update = datetime.now()
            
            logger.info(f"Updated {len(self.positions)} positions")
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    def _convert_positions(self, raw_positions: List) -> Dict[str, RealPosition]:
        """Convert IBKR positions to internal format"""
        converted = {}
        
        for pos in raw_positions:
            if pos.position == 0:  # Skip zero positions
                continue
                
            symbol = pos.contract.symbol
            position = RealPosition(
                symbol=symbol,
                asset_class=self._get_asset_class(pos.contract),
                quantity=pos.position,
                avg_cost=pos.avgCost,
                market_value=pos.marketValue,
                unrealized_pnl=pos.unrealizedPNL,
                realized_pnl=pos.realizedPNL,
                currency=pos.contract.currency,
                exchange=pos.contract.exchange,
                contract_info={
                    'secType': pos.contract.secType,
                    'conId': pos.contract.conId,
                    'localSymbol': pos.contract.localSymbol
                }
            )
            converted[symbol] = position
            
        return converted
    
    def _get_asset_class(self, contract) -> str:
        """Determine asset class from contract"""
        sec_type_map = {
            'STK': 'stocks',
            'CASH': 'forex', 
            'FUT': 'futures',
            'OPT': 'options',
            'CRYPTO': 'crypto',
            'BOND': 'bonds',
            'CMDTY': 'commodities'
        }
        
        asset_class = sec_type_map.get(contract.secType, 'other')
        
        # Special handling for commodity futures
        if asset_class == 'futures' and contract.symbol in ['GC', 'SI', 'CL', 'NG', 'HG']:
            asset_class = 'commodities'
            
        return asset_class
    
    def get_positions_by_asset_class(self, asset_class: str = None) -> List[RealPosition]:
        """Get positions filtered by asset class"""
        if not asset_class:
            return list(self.positions.values())
        
        return [pos for pos in self.positions.values() 
                if pos.asset_class == asset_class]
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        summary = {
            'total_market_value': 0,
            'total_unrealized_pnl': 0,
            'total_realized_pnl': 0,
            'position_count': len(self.positions),
            'by_asset_class': {},
            'by_currency': {},
            'last_updated': self.last_update.isoformat() if self.last_update else None
        }
        
        # Calculate totals and breakdowns
        for position in self.positions.values():
            # Overall totals
            summary['total_market_value'] += position.market_value
            summary['total_unrealized_pnl'] += position.unrealized_pnl
            summary['total_realized_pnl'] += position.realized_pnl
            
            # By asset class
            asset_class = position.asset_class
            if asset_class not in summary['by_asset_class']:
                summary['by_asset_class'][asset_class] = {
                    'market_value': 0, 'unrealized_pnl': 0, 'count': 0, 'symbols': []
                }
            
            ac_summary = summary['by_asset_class'][asset_class]
            ac_summary['market_value'] += position.market_value
            ac_summary['unrealized_pnl'] += position.unrealized_pnl
            ac_summary['count'] += 1
            ac_summary['symbols'].append(position.symbol)
            
            # By currency
            currency = position.currency
            if currency not in summary['by_currency']:
                summary['by_currency'][currency] = {'market_value': 0, 'count': 0}
            
            summary['by_currency'][currency]['market_value'] += position.market_value
            summary['by_currency'][currency]['count'] += 1
        
        return summary
    
    def update_strategy_performance(self, strategy_name: str, trade_result: Dict):
        """Update strategy performance metrics"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = StrategyPerformance(
                name=strategy_name,
                signals_generated=0,
                trades_executed=0,
                total_pnl=0.0,
                win_rate=0.0,
                avg_trade_duration=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                active=True
            )
        
        perf = self.strategy_performance[strategy_name]
        perf.trades_executed += 1
        perf.total_pnl += trade_result.get('pnl', 0)
        
        # Update win rate
        if trade_result.get('pnl', 0) > 0:
            wins = (perf.win_rate * (perf.trades_executed - 1) + 1) / perf.trades_executed
            perf.win_rate = wins
        else:
            perf.win_rate = (perf.win_rate * (perf.trades_executed - 1)) / perf.trades_executed
    
    def get_strategy_performance_list(self) -> List[Dict]:
        """Get strategy performance as list for dashboard"""
        return [
            {
                'name': perf.name,
                'signals': perf.signals_generated,
                'pnl': perf.total_pnl,
                'win_rate': perf.win_rate,
                'active': perf.active,
                'status': 'Running' if perf.active else 'Stopped'
            }
            for perf in self.strategy_performance.values()
        ]

# Global position manager instance
_position_manager = None

def get_position_manager():
    """Get global position manager instance"""
    global _position_manager
    if _position_manager is None:
        _position_manager = RealPositionManager()
    return _position_manager

def get_real_positions() -> List[Dict]:
    """Get real positions for dashboard"""
    try:
        manager = get_position_manager()
        positions = manager.get_positions_by_asset_class()
        
        return [
            {
                'symbol': pos.symbol,
                'quantity': int(pos.quantity),
                'price': pos.avg_cost,
                'pnl': pos.unrealized_pnl,
                'pnl_pct': pos.unrealized_pnl / max(abs(pos.market_value), 1) if pos.market_value else 0,
                'asset_class': pos.asset_class,
                'currency': pos.currency
            }
            for pos in positions
        ]
    except Exception as e:
        logger.error(f"Error getting real positions: {e}")
        return []

def get_strategy_performance() -> List[Dict]:
    """Get strategy performance for dashboard"""
    try:
        manager = get_position_manager()
        return manager.get_strategy_performance_list()
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        return []
