#!/usr/bin/env python3
"""
Historical Data Backtesting Framework
Download historical data and run comprehensive backtests

Usage: python historical_backtest.py
"""

import asyncio
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class HistoricalDataManager:
    """Manages historical data download and storage"""
    
    def __init__(self, data_dir: str = "historical_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_symbol_data(self, symbol: str, start_date: str, end_date: str, 
                           interval: str = "1d") -> Optional[pd.DataFrame]:
        """Download historical data for a symbol"""
        try:
            # For IBKR symbols, convert format
            yf_symbol = self._convert_to_yf_symbol(symbol)
            
            print(f"📥 Downloading {symbol} ({yf_symbol}) from {start_date} to {end_date}")
            
            data = yf.download(yf_symbol, start=start_date, end=end_date, 
                             interval=interval, progress=False)
            
            if data.empty:
                print(f"❌ No data for {symbol}")
                return None
            
            # Clean and prepare data
            data = data.dropna()
            data.columns = [col.lower() for col in data.columns]
            
            # Save to file
            filename = self.data_dir / f"{symbol}_{start_date}_{end_date}_{interval}.csv"
            data.to_csv(filename)
            
            print(f"✅ {symbol}: {len(data)} bars downloaded")
            return data
            
        except Exception as e:
            print(f"❌ Failed to download {symbol}: {e}")
            return None
    
    def _convert_to_yf_symbol(self, symbol: str) -> str:
        """Convert IBKR symbol to Yahoo Finance format"""
        # Forex conversions
        forex_map = {
            'EUR.USD': 'EURUSD=X',
            'GBP.USD': 'GBPUSD=X',
            'USD.JPY': 'USDJPY=X',
            'AUD.USD': 'AUDUSD=X',
            'USD.CHF': 'USDCHF=X',
            'USD.CAD': 'USDCAD=X',
            'NZD.USD': 'NZDUSD=X'
        }
        
        # Futures conversions (using ETFs as proxies)
        futures_map = {
            'ES': '^GSPC',  # S&P 500
            'NQ': '^IXIC',  # NASDAQ
            'RTY': '^RUT',  # Russell 2000
            'GC': 'GC=F',   # Gold futures
            'CL': 'CL=F',   # Crude oil futures
            'NG': 'NG=F',   # Natural gas futures
        }
        
        # Check mappings
        if symbol in forex_map:
            return forex_map[symbol]
        elif symbol in futures_map:
            return futures_map[symbol]
        else:
            return symbol  # Assume it's a stock symbol
    
    def load_cached_data(self, symbol: str, start_date: str, end_date: str, 
                        interval: str = "1d") -> Optional[pd.DataFrame]:
        """Load cached historical data"""
        filename = self.data_dir / f"{symbol}_{start_date}_{end_date}_{interval}.csv"
        
        if filename.exists():
            try:
                data = pd.read_csv(filename, index_col=0, parse_dates=True)
                print(f"📁 Loaded cached data for {symbol}: {len(data)} bars")
                return data
            except Exception as e:
                print(f"⚠️ Could not load cached data for {symbol}: {e}")
        
        return None

class StrategyBacktester:
    """Backtests trading strategies on historical data"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        
    def backtest_moving_average_strategy(self, data: pd.DataFrame, 
                                       fast_period: int = 10, slow_period: int = 30,
                                       symbol: str = "UNKNOWN") -> Dict:
        """Backtest moving average crossover strategy"""
        df = data.copy()
        
        # Calculate moving averages
        df['ma_fast'] = df['close'].rolling(fast_period).mean()
        df['ma_slow'] = df['close'].rolling(slow_period).mean()
        
        # Generate signals
        df['signal'] = 0
        df['signal'][slow_period:] = np.where(
            df['ma_fast'][slow_period:] > df['ma_slow'][slow_period:], 1, 0
        )
        df['position'] = df['signal'].diff()
        
        # Calculate returns
        trades = self._extract_trades(df, 'position')
        metrics = self._calculate_metrics(trades, df, symbol)
        
        return {
            'strategy': 'moving_average',
            'symbol': symbol,
            'parameters': {'fast_period': fast_period, 'slow_period': slow_period},
            'metrics': metrics,
            'trades': trades,
            'equity_curve': self._calculate_equity_curve(trades)
        }
    
    def backtest_rsi_strategy(self, data: pd.DataFrame, rsi_period: int = 14,
                            oversold: int = 30, overbought: int = 70,
                            symbol: str = "UNKNOWN") -> Dict:
        """Backtest RSI mean reversion strategy"""
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], rsi_period)
        
        # Generate signals
        df['signal'] = 0
        df['signal'] = np.where(df['rsi'] < oversold, 1, 0)  # Buy on oversold
        df['signal'] = np.where(df['rsi'] > overbought, -1, df['signal'])  # Sell on overbought
        df['position'] = df['signal'].diff()
        
        # Calculate returns
        trades = self._extract_trades(df, 'position')
        metrics = self._calculate_metrics(trades, df, symbol)
        
        return {
            'strategy': 'rsi_mean_reversion',
            'symbol': symbol,
            'parameters': {'rsi_period': rsi_period, 'oversold': oversold, 'overbought': overbought},
            'metrics': metrics,
            'trades': trades,
            'equity_curve': self._calculate_equity_curve(trades)
        }
    
    def backtest_bollinger_bands(self, data: pd.DataFrame, period: int = 20, 
                                std_dev: float = 2, symbol: str = "UNKNOWN") -> Dict:
        """Backtest Bollinger Bands strategy"""
        df = data.copy()
        
        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(period).mean()
        df['std'] = df['close'].rolling(period).std()
        df['upper_band'] = df['sma'] + (std_dev * df['std'])
        df['lower_band'] = df['sma'] - (std_dev * df['std'])
        
        # Generate signals (breakout strategy)
        df['signal'] = 0
        df['signal'] = np.where(df['close'] > df['upper_band'], 1, 0)  # Buy on upper breakout
        df['signal'] = np.where(df['close'] < df['lower_band'], -1, df['signal'])  # Sell on lower breakout
        df['position'] = df['signal'].diff()
        
        # Calculate returns
        trades = self._extract_trades(df, 'position')
        metrics = self._calculate_metrics(trades, df, symbol)
        
        return {
            'strategy': 'bollinger_bands',
            'symbol': symbol,
            'parameters': {'period': period, 'std_dev': std_dev},
            'metrics': metrics,
            'trades': trades,
            'equity_curve': self._calculate_equity_curve(trades)
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _extract_trades(self, df: pd.DataFrame, position_col: str) -> List[Dict]:
        """Extract individual trades from position changes"""
        trades = []
        
        buy_signals = df[df[position_col] == 1]
        sell_signals = df[df[position_col] == -1]
        
        min_trades = min(len(buy_signals), len(sell_signals))
        
        for i in range(min_trades):
            buy_date = buy_signals.index[i]
            buy_price = buy_signals.iloc[i]['close']
            
            sell_date = sell_signals.index[i]
            sell_price = sell_signals.iloc[i]['close']
            
            # Calculate trade metrics
            trade_return = (sell_price - buy_price) / buy_price
            trade_days = (sell_date - buy_date).days
            
            trades.append({
                'entry_date': buy_date,
                'exit_date': sell_date,
                'entry_price': buy_price,
                'exit_price': sell_price,
                'return': trade_return,
                'return_pct': trade_return * 100,
                'days_held': trade_days,
                'profit_loss': trade_return * self.initial_capital
            })
        
        return trades
    
    def _calculate_metrics(self, trades: List[Dict], df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate comprehensive strategy metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        returns = [trade['return'] for trade in trades]
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_return = np.prod([1 + r for r in returns]) - 1
        avg_trade_return = np.mean(returns) if returns else 0
        
        # Risk metrics
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Drawdown calculation
        equity_curve = self._calculate_equity_curve(trades)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'avg_trade_return_pct': avg_trade_return * 100,
            'best_trade': max(returns) if returns else 0,
            'worst_trade': min(returns) if returns else 0,
            'avg_days_held': np.mean([t['days_held'] for t in trades]) if trades else 0,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def _calculate_equity_curve(self, trades: List[Dict]) -> List[float]:
        """Calculate equity curve"""
        equity = [self.initial_capital]
        
        for trade in trades:
            new_equity = equity[-1] * (1 + trade['return'])
            equity.append(new_equity)
        
        return equity
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd

class BacktestRunner:
    """Runs comprehensive backtests across multiple symbols and strategies"""
    
    def __init__(self):
        self.data_manager = HistoricalDataManager()
        self.backtester = StrategyBacktester()
        self.results = {}
    
    async def run_comprehensive_backtest(self, symbols: List[str], 
                                       start_date: str = "2020-01-01",
                                       end_date: str = "2024-01-01") -> Dict:
        """Run backtest across multiple symbols and strategies"""
        print(f"🚀 Starting Comprehensive Backtest")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"📊 Symbols: {symbols}")
        print("=" * 60)
        
        all_results = {}
        
        for symbol in symbols:
            print(f"\n📈 Processing {symbol}...")
            
            # Download/load data
            data = await self._get_historical_data(symbol, start_date, end_date)
            if data is None:
                continue
            
            # Run all strategies
            symbol_results = {}
            
            # Moving Average Strategy
            ma_result = self.backtester.backtest_moving_average_strategy(data, symbol=symbol)
            symbol_results['moving_average'] = ma_result
            
            # RSI Strategy
            rsi_result = self.backtester.backtest_rsi_strategy(data, symbol=symbol)
            symbol_results['rsi_mean_reversion'] = rsi_result
            
            # Bollinger Bands Strategy
            bb_result = self.backtester.backtest_bollinger_bands(data, symbol=symbol)
            symbol_results['bollinger_bands'] = bb_result
            
            all_results[symbol] = symbol_results
            
            # Print summary for this symbol
            self._print_symbol_summary(symbol, symbol_results)
        
        # Generate comprehensive report
        self._generate_comprehensive_report(all_results, start_date, end_date)
        
        # Save results
        self._save_backtest_results(all_results, start_date, end_date)
        
        return all_results
    
    async def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get historical data, using cache if available"""
        # Try to load cached data first
        data = self.data_manager.load_cached_data(symbol, start_date, end_date)
        
        if data is None:
            # Download fresh data
            data = self.data_manager.download_symbol_data(symbol, start_date, end_date)
        
        return data
    
    def _print_symbol_summary(self, symbol: str, results: Dict):
        """Print summary for a symbol"""
        print(f"\n📊 {symbol} Results:")
        
        for strategy_name, result in results.items():
            metrics = result['metrics']
            print(f"   {strategy_name:20}: "
                  f"Return {metrics['total_return_pct']:6.1f}% | "
                  f"Sharpe {metrics['sharpe_ratio']:5.2f} | "
                  f"Trades {metrics['total_trades']:3d} | "
                  f"Win% {metrics['win_rate']*100:5.1f}%")
    
    def _generate_comprehensive_report(self, all_results: Dict, start_date: str, end_date: str):
        """Generate comprehensive backtest report"""
        print(f"\n📋 COMPREHENSIVE BACKTEST REPORT")
        print(f"📅 Period: {start_date} to {end_date}")
        print("=" * 80)
        
        # Strategy performance summary
        strategy_summary = {}
        
        for symbol, symbol_results in all_results.items():
            for strategy_name, result in symbol_results.items():
                if strategy_name not in strategy_summary:
                    strategy_summary[strategy_name] = []
                strategy_summary[strategy_name].append(result['metrics'])
        
        print(f"\n🎯 STRATEGY PERFORMANCE SUMMARY:")
        print(f"{'Strategy':<20} {'Avg Return':<12} {'Avg Sharpe':<12} {'Avg Trades':<12} {'Avg Win%':<12}")
        print("-" * 80)
        
        for strategy_name, metrics_list in strategy_summary.items():
            avg_return = np.mean([m['total_return_pct'] for m in metrics_list])
            avg_sharpe = np.mean([m['sharpe_ratio'] for m in metrics_list])
            avg_trades = np.mean([m['total_trades'] for m in metrics_list])
            avg_winrate = np.mean([m['win_rate'] for m in metrics_list]) * 100
            
            print(f"{strategy_name:<20} {avg_return:>10.1f}% {avg_sharpe:>10.2f} "
                  f"{avg_trades:>10.1f} {avg_winrate:>10.1f}%")
        
        # Best performing combinations
        print(f"\n🏆 TOP PERFORMING COMBINATIONS:")
        all_combinations = []
        
        for symbol, symbol_results in all_results.items():
            for strategy_name, result in symbol_results.items():
                all_combinations.append({
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'return': result['metrics']['total_return_pct'],
                    'sharpe': result['metrics']['sharpe_ratio'],
                    'trades': result['metrics']['total_trades'],
                    'win_rate': result['metrics']['win_rate']
                })
        
        # Sort by return
        top_by_return = sorted(all_combinations, key=lambda x: x['return'], reverse=True)[:10]
        
        print(f"{'Rank':<5} {'Symbol':<8} {'Strategy':<20} {'Return':<10} {'Sharpe':<8}")
        print("-" * 60)
        for i, combo in enumerate(top_by_return, 1):
            print(f"{i:<5} {combo['symbol']:<8} {combo['strategy']:<20} "
                  f"{combo['return']:>8.1f}% {combo['sharpe']:>6.2f}")
    
    def _save_backtest_results(self, results: Dict, start_date: str, end_date: str):
        """Save backtest results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_filename = f"backtest_results_{start_date}_{end_date}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2, default=str)
        
        # Save CSV summary
        csv_filename = f"backtest_summary_{start_date}_{end_date}_{timestamp}.csv"
        summary_data = []
        
        for symbol, symbol_results in results.items():
            for strategy_name, result in symbol_results.items():
                metrics = result['metrics']
                summary_data.append({
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'total_return_pct': metrics['total_return_pct'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown_pct': metrics['max_drawdown_pct'],
                    'total_trades': metrics['total_trades'],
                    'win_rate': metrics['win_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'avg_trade_return_pct': metrics['avg_trade_return_pct']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_filename, index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   📄 {json_filename} (detailed results)")
        print(f"   📄 {csv_filename} (summary)")
    
    def _convert_for_json(self, obj):
        """Convert numpy types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj

# CLI Commands for main.py integration
def add_backtest_commands_to_main():
    """
    Add these commands to your main.py file:
    """
    
    example_commands = '''
# Add these imports to main.py:
from historical_backtest import BacktestRunner, HistoricalDataManager

# Add these commands to your main.py:

@app.command()
def discover_symbols():
    """Discover available symbols in IBKR"""
    setup_logging("INFO")
    
    print("🔍 Starting symbol discovery...")
    
    import subprocess
    try:
        result = subprocess.run([sys.executable, 'symbol_discovery.py'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        print(f"❌ Discovery failed: {e}")

@app.command()
def backtest_symbols(
    symbols: str = typer.Option("AAPL,MSFT,SPY", help="Comma-separated symbols"),
    start_date: str = typer.Option("2020-01-01", help="Start date YYYY-MM-DD"),
    end_date: str = typer.Option("2024-01-01", help="End date YYYY-MM-DD"),
    strategies: str = typer.Option("all", help="Strategies to test: all, ma, rsi, bb")
):
    """Run comprehensive backtests on historical data"""
    setup_logging("INFO")
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    async def run_backtest():
        runner = BacktestRunner()
        results = await runner.run_comprehensive_backtest(symbol_list, start_date, end_date)
        return results
    
    asyncio.run(run_backtest())

@app.command()
def download_data(
    symbols: str = typer.Option("AAPL,MSFT,SPY", help="Symbols to download"),
    start_date: str = typer.Option("2020-01-01", help="Start date"),
    end_date: str = typer.Option("2024-01-01", help="End date"),
    interval: str = typer.Option("1d", help="Data interval: 1d, 1h, 5m")
):
    """Download historical data for symbols"""
    setup_logging("INFO")
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    data_manager = HistoricalDataManager()
    
    print(f"📥 Downloading data for {len(symbol_list)} symbols...")
    
    for symbol in symbol_list:
        data = data_manager.download_symbol_data(symbol, start_date, end_date, interval)
        if data is not None:
            print(f"✅ {symbol}: {len(data)} bars downloaded")
        else:
            print(f"❌ {symbol}: Download failed")

@app.command() 
def analyze_best_symbols(
    min_return: float = typer.Option(10.0, help="Minimum return percentage"),
    min_sharpe: float = typer.Option(1.0, help="Minimum Sharpe ratio"),
    max_drawdown: float = typer.Option(20.0, help="Maximum drawdown percentage")
):
    """Analyze and find best performing symbols from backtest results"""
    setup_logging("INFO")
    
    print("🔍 Analyzing backtest results for best symbols...")
    
    # Look for recent backtest results
    import glob
    result_files = glob.glob("backtest_summary_*.csv")
    
    if not result_files:
        print("❌ No backtest results found. Run backtests first.")
        return
    
    # Use most recent results
    latest_file = max(result_files, key=os.path.getctime)
    print(f"📄 Using results from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Filter by criteria
    filtered = df[
        (df['total_return_pct'] >= min_return) &
        (df['sharpe_ratio'] >= min_sharpe) &
        (df['max_drawdown_pct'] <= max_drawdown)
    ]
    
    if filtered.empty:
        print(f"❌ No symbols meet the criteria:")
        print(f"   Return >= {min_return}%")
        print(f"   Sharpe >= {min_sharpe}")
        print(f"   Max DD <= {max_drawdown}%")
        return
    
    # Sort by return
    best_symbols = filtered.sort_values('total_return_pct', ascending=False)
    
    print(f"🏆 Best performing symbols ({len(best_symbols)} found):")
    print(f"{'Symbol':<8} {'Strategy':<20} {'Return':<10} {'Sharpe':<8} {'MaxDD':<8}")
    print("-" * 60)
    
    for _, row in best_symbols.head(20).iterrows():
        print(f"{row['symbol']:<8} {row['strategy']:<20} "
              f"{row['total_return_pct']:>8.1f}% {row['sharpe_ratio']:>6.2f} "
              f"{row['max_drawdown_pct']:>6.1f}%")
    
    # Save filtered results
    output_file = f"best_symbols_{min_return}ret_{min_sharpe}sharpe.csv"
    best_symbols.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")
'''
    
    return example_commands

# Main function for standalone usage
async def main():
    """Main function for standalone backtesting"""
    print("📊 Historical Data Backtesting Framework")
    print("=" * 60)
    
    # Default symbols for testing
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
    
    print(f"🧪 Running test backtest with symbols: {test_symbols}")
    
    runner = BacktestRunner()
    results = await runner.run_comprehensive_backtest(
        symbols=test_symbols,
        start_date="2022-01-01",
        end_date="2024-01-01"
    )
    
    print(f"\n✅ Backtest completed!")
    print(f"📊 Tested {len(results)} symbols with 3 strategies each")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Backtest cancelled")
    except Exception as e:
        print(f"❌ Backtest failed: {e}")