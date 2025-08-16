# strategies/__init__.py - Complete implementation
from typing import Optional, Dict, Any, List
from core.interfaces import IStrategy, IEventBus, MarketData, Signal, OrderAction
from loguru import logger
import numpy as np
from datetime import datetime


# Import base strategy
from .base import BaseStrategy

def create_strategy(strategy_type: str, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]) -> IStrategy:
    """Factory function to create strategy instances"""
    
    # Map strategy names from config to actual classes
    strategy_map = {
        'ma': MovingAverageStrategy,
        'bollinger': BollingerBandsStrategy,
        'momentum': MomentumStrategy,
        'rsi': RSIMeanReversionStrategy,
        'lstm': LSTMStrategy,
        'rf': RandomForestStrategy,
        # Add the new strategies
        'commodity': CommoditySeasonalityStrategy,
        'forex': ForexCarryTradeStrategy,
        'multi': MultiMarketMomentumStrategy,
    }
    
    if strategy_type not in strategy_map:
        # For strategies not yet implemented, create a momentum strategy as fallback
        logger.warning(f"Strategy type '{strategy_type}' not found, using MomentumStrategy as fallback")
        return MomentumStrategy(strategy_id, event_bus, config)
    
    strategy_class = strategy_map[strategy_type]
    return strategy_class(strategy_id, event_bus, config)


class CommoditySeasonalityStrategy(BaseStrategy):
    """Commodity seasonality strategy"""
    
    def __init__(self, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]):
        super().__init__(strategy_id, event_bus, config)
        self.markets = config.get('markets', {})
        self.commodity_symbols = self.markets.get('commodities', ['GC', 'CL', 'NG'])
        self.symbols = self.commodity_symbols  # Set symbols for base class
        self.seasonal_patterns = config.get('seasonal_patterns', True)
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Generate commodity seasonality signal"""
        if data.symbol not in self.commodity_symbols:
            return None
            
        prices = self.get_price_array(data.symbol)
        if len(prices) < 20:
            return None
        
        # Simple seasonality logic based on current month
        current_month = datetime.now().month
        
        # Example: Energy commodities tend to be stronger in winter
        if data.symbol in ['NG', 'HO'] and current_month in [11, 12, 1, 2]:
            # Winter months - bullish for heating fuels
            confidence = 0.6
            action = OrderAction.BUY
        elif data.symbol in ['CL'] and current_month in [5, 6, 7, 8]:
            # Driving season - bullish for crude
            confidence = 0.7
            action = OrderAction.BUY
        elif data.symbol in ['ZC', 'ZS'] and current_month in [6, 7, 8]:
            # Growing season concerns
            recent_volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
            if recent_volatility > 0.02:  # High volatility
                confidence = 0.5
                action = OrderAction.BUY
            else:
                return None
        else:
            return None
        
        return Signal(
            symbol=data.symbol,
            action=action,
            confidence=confidence,
            strategy_id=self.strategy_id,
            timestamp=datetime.now(),
            metadata={
                'seasonal_month': current_month,
                'commodity_type': self._get_commodity_type(data.symbol),
                'price': data.price
            }
        )
    
    def _get_commodity_type(self, symbol: str) -> str:
        """Get commodity type"""
        energy = ['CL', 'NG', 'HO', 'RB']
        metals = ['GC', 'SI', 'HG', 'PL']
        agriculture = ['ZC', 'ZS', 'ZW', 'CT', 'SB']
        
        if symbol in energy:
            return 'energy'
        elif symbol in metals:
            return 'metals'
        elif symbol in agriculture:
            return 'agriculture'
        else:
            return 'other'


class ForexCarryTradeStrategy(BaseStrategy):
    """Forex carry trade strategy"""
    
    def __init__(self, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]):
        super().__init__(strategy_id, event_bus, config)
        self.markets = config.get('markets', {})
        self.forex_symbols = self.markets.get('forex', ['EUR.USD', 'AUD.USD', 'USD.JPY'])
        self.symbols = self.forex_symbols
        self.interest_rate_threshold = config.get('interest_rate_threshold', 0.02)
        self.hold_period_days = config.get('hold_period_days', 30)
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Generate forex carry trade signal"""
        if data.symbol not in self.forex_symbols:
            return None
            
        prices = self.get_price_array(data.symbol)
        if len(prices) < 20:
            return None
        
        # Simple carry trade logic based on currency pair
        base_currency = data.symbol.split('.')[0] if '.' in data.symbol else data.symbol[:3]
        quote_currency = data.symbol.split('.')[1] if '.' in data.symbol else data.symbol[3:]
        
        # Simplified interest rate differentials (normally would get from economic data)
        high_yield_currencies = ['AUD', 'NZD', 'CAD']
        low_yield_currencies = ['JPY', 'CHF', 'EUR']
        
        # Calculate trend strength
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-20:])
        trend_strength = (short_ma - long_ma) / long_ma
        
        signal = None
        
        if base_currency in high_yield_currencies and quote_currency in low_yield_currencies:
            # High yield vs low yield - potential carry trade
            if trend_strength > 0.001:  # Uptrend supports carry
                confidence = min(0.8, trend_strength * 100)
                signal = Signal(
                    symbol=data.symbol,
                    action=OrderAction.BUY,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    metadata={
                        'carry_type': 'positive',
                        'base_currency': base_currency,
                        'quote_currency': quote_currency,
                        'trend_strength': trend_strength
                    }
                )
        elif base_currency in low_yield_currencies and quote_currency in high_yield_currencies:
            # Low yield vs high yield - potential reverse carry
            if trend_strength < -0.001:  # Downtrend supports reverse carry
                confidence = min(0.8, abs(trend_strength) * 100)
                signal = Signal(
                    symbol=data.symbol,
                    action=OrderAction.SELL,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    metadata={
                        'carry_type': 'negative',
                        'base_currency': base_currency,
                        'quote_currency': quote_currency,
                        'trend_strength': trend_strength
                    }
                )
        
        return signal


class MultiMarketMomentumStrategy(BaseStrategy):
    """Multi-market momentum strategy"""
    
    def __init__(self, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]):
        super().__init__(strategy_id, event_bus, config)
        self.markets = config.get('markets', {})
        self.momentum_threshold = config.get('momentum_threshold', 0.02)
        
        # Collect all symbols from all markets
        all_symbols = []
        for market_name, symbols in self.markets.items():
            if isinstance(symbols, list):
                all_symbols.extend(symbols)
        
        self.symbols = all_symbols
        self.lookback_period = config.get('lookback_period', 20)
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Generate multi-market momentum signal"""
        prices = self.get_price_array(data.symbol)
        if len(prices) < self.lookback_period:
            return None
        
        # Calculate momentum
        current_price = prices[-1]
        lookback_price = prices[-self.lookback_period]
        momentum = (current_price - lookback_price) / lookback_price
        
        # Determine which market this symbol belongs to
        market_type = self._get_market_type(data.symbol)
        
        # Calculate volatility for confidence adjustment
        recent_returns = np.diff(prices[-10:]) / prices[-10:-1]
        volatility = np.std(recent_returns)
        
        # Generate signal based on momentum threshold
        signal = None
        
        if momentum > self.momentum_threshold:
            # Strong upward momentum
            confidence = min(0.9, momentum / self.momentum_threshold * 0.5)
            # Adjust confidence based on volatility
            if volatility < 0.02:  # Low volatility = higher confidence
                confidence *= 1.2
            
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.BUY,
                confidence=min(confidence, 1.0),
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={
                    'momentum': momentum,
                    'market_type': market_type,
                    'volatility': volatility,
                    'lookback_period': self.lookback_period
                }
            )
        
        elif momentum < -self.momentum_threshold:
            # Strong downward momentum
            confidence = min(0.9, abs(momentum) / self.momentum_threshold * 0.5)
            # Adjust confidence based on volatility
            if volatility < 0.02:
                confidence *= 1.2
            
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.SELL,
                confidence=min(confidence, 1.0),
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={
                    'momentum': momentum,
                    'market_type': market_type,
                    'volatility': volatility,
                    'lookback_period': self.lookback_period
                }
            )
        
        return signal
    
    def _get_market_type(self, symbol: str) -> str:
        """Determine which market the symbol belongs to"""
        for market_name, symbols in self.markets.items():
            if isinstance(symbols, list) and symbol in symbols:
                return market_name
        return 'unknown'
    

class MovingAverageStrategy(BaseStrategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]):
        super().__init__(strategy_id, event_bus, config)
        self.fast_period = config.get('fast_period', 10)
        self.slow_period = config.get('slow_period', 30)
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']  # Default symbols
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Generate MA crossover signal"""
        prices = self.get_price_array(data.symbol)
        
        if len(prices) < self.slow_period:
            return None
        
        # Calculate moving averages
        fast_ma = np.mean(prices[-self.fast_period:])
        slow_ma = np.mean(prices[-self.slow_period:])
        
        # Previous MA values for crossover detection
        if len(prices) < self.slow_period + 1:
            return None
            
        prev_fast_ma = np.mean(prices[-(self.fast_period + 1):-1])
        prev_slow_ma = np.mean(prices[-(self.slow_period + 1):-1])
        
        signal = None
        
        # Bullish crossover: fast MA crosses above slow MA
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            confidence = min((fast_ma - slow_ma) / slow_ma * 10, 1.0)
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.BUY,
                confidence=confidence,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma, 'price': data.price}
            )
        
        # Bearish crossover: fast MA crosses below slow MA
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            confidence = min((slow_ma - fast_ma) / slow_ma * 10, 1.0)
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.SELL,
                confidence=confidence,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma, 'price': data.price}
            )
        
        return signal


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Strategy"""
    
    def __init__(self, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]):
        super().__init__(strategy_id, event_bus, config)
        self.period = config.get('period', 20)
        self.std_dev = config.get('std_dev', 2)
        self.mode = config.get('mode', 'breakout')  # 'breakout' or 'reversion'
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Generate Bollinger Bands signal"""
        prices = self.get_price_array(data.symbol)
        
        if len(prices) < self.period:
            return None
        
        # Calculate Bollinger Bands
        sma = np.mean(prices[-self.period:])
        std = np.std(prices[-self.period:])
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)
        
        current_price = data.price
        signal = None
        
        if self.mode == 'breakout':
            # Breakout strategy: buy on upper band break, sell on lower band break
            if current_price > upper_band:
                confidence = min((current_price - upper_band) / upper_band * 5, 1.0)
                signal = Signal(
                    symbol=data.symbol,
                    action=OrderAction.BUY,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    metadata={'upper_band': upper_band, 'lower_band': lower_band, 'sma': sma}
                )
            elif current_price < lower_band:
                confidence = min((lower_band - current_price) / lower_band * 5, 1.0)
                signal = Signal(
                    symbol=data.symbol,
                    action=OrderAction.SELL,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    metadata={'upper_band': upper_band, 'lower_band': lower_band, 'sma': sma}
                )
        
        elif self.mode == 'reversion':
            # Mean reversion: sell at upper band, buy at lower band
            if current_price > upper_band:
                confidence = min((current_price - upper_band) / upper_band * 5, 1.0)
                signal = Signal(
                    symbol=data.symbol,
                    action=OrderAction.SELL,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    metadata={'upper_band': upper_band, 'lower_band': lower_band, 'sma': sma}
                )
            elif current_price < lower_band:
                confidence = min((lower_band - current_price) / lower_band * 5, 1.0)
                signal = Signal(
                    symbol=data.symbol,
                    action=OrderAction.BUY,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    timestamp=datetime.now(),
                    metadata={'upper_band': upper_band, 'lower_band': lower_band, 'sma': sma}
                )
        
        return signal


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy"""
    
    def __init__(self, strategy_id: str, event_bus: IEventBus, config: Dict[str, Any]):
        super().__init__(strategy_id, event_bus, config)
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_threshold = config.get('oversold_threshold', 30)
        self.overbought_threshold = config.get('overbought_threshold', 70)
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        """Generate RSI mean reversion signal"""
        prices = self.get_price_array(data.symbol)
        
        if len(prices) < self.rsi_period + 1:
            return None
        
        # Calculate RSI
        rsi = self._calculate_rsi(prices)
        signal = None
        
        # Oversold condition - potential buy signal
        if rsi < self.oversold_threshold:
            confidence = (self.oversold_threshold - rsi) / self.oversold_threshold
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.BUY,
                confidence=confidence,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={'rsi': rsi, 'price': data.price}
            )
        
        # Overbought condition - potential sell signal
        elif rsi > self.overbought_threshold:
            confidence = (rsi - self.overbought_threshold) / (100 - self.overbought_threshold)
            signal = Signal(
                symbol=data.symbol,
                action=OrderAction.SELL,
                confidence=confidence,
                strategy_id=self.strategy_id,
                timestamp=datetime.now(),
                metadata={'rsi': rsi, 'price': data.price}
            )
        
        return signal
    
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


# Placeholder strategies for ML models
class LSTMStrategy(BaseStrategy):
    """LSTM Strategy Placeholder"""
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        logger.info(f"LSTM strategy processing {data.symbol} - not implemented yet")
        return None


class RandomForestStrategy(BaseStrategy):
    """Random Forest Strategy Placeholder"""
    
    async def _generate_signal(self, data: MarketData) -> Optional[Signal]:
        logger.info(f"Random Forest strategy processing {data.symbol} - not implemented yet")
        return None


# Import the momentum strategy from the existing file
from .conventional.momentum import MomentumStrategy

__all__ = ['create_strategy']