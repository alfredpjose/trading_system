#!/usr/bin/env python3
"""
IBKR Symbol Discovery Tool
Discovers available symbols across all asset classes and tests their availability

Usage: python symbol_discovery.py
"""

import asyncio
import os
import sys
import time
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class SymbolDiscovery(EWrapper, EClient):
    """IBKR Symbol Discovery Tool"""
    
    def __init__(self):
        EClient.__init__(self, self)
        
        self.host = os.getenv('IBKR_HOST', '127.0.0.1')
        self.port = int(os.getenv('IBKR_PORT', '7497'))
        self.client_id = int(os.getenv('IBKR_CLIENT_ID', '3000')) + 500
        
        self.connected = False
        self.contract_details = {}
        self.market_data_results = {}
        self.current_request = None
        self.request_complete = False
        
    def connectAck(self):
        self.connected = True
        print("✅ Connected to IBKR for symbol discovery")
    
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode == 200:  # No security definition
            print(f"❌ {self.current_request}: Not available")
            self.request_complete = True
        elif errorCode in [2104, 2106, 2158, 2119]:
            pass  # Ignore info messages
        else:
            print(f"⚠️  {errorCode}: {errorString}")
    
    def contractDetails(self, reqId, contractDetails):
        """Handle contract details response"""
        contract = contractDetails.contract
        symbol = contract.symbol
        
        self.contract_details[symbol] = {
            'symbol': symbol,
            'secType': contract.secType,
            'exchange': contract.exchange,
            'currency': contract.currency,
            'primaryExchange': contract.primaryExchange,
            'tradingClass': contract.tradingClass,
            'minTick': contractDetails.minTick,
            'marketName': contractDetails.marketName,
            'longName': contractDetails.longName,
            'category': contractDetails.category,
            'subcategory': contractDetails.subcategory,
            'available': True
        }
        
        print(f"✅ {symbol}: {contractDetails.longName} ({contract.exchange})")
    
    def contractDetailsEnd(self, reqId):
        """Contract details request completed"""
        self.request_complete = True
    
    def tickPrice(self, reqId, tickType, price, attrib):
        """Handle market data for availability test"""
        if reqId in self.market_data_results:
            symbol = self.market_data_results[reqId]['symbol']
            if tickType == 4:  # Last price
                self.market_data_results[reqId]['last_price'] = price
                self.market_data_results[reqId]['data_available'] = True
    
    async def connect_to_ibkr(self):
        """Connect to IBKR"""
        await asyncio.get_event_loop().run_in_executor(
            None, self.connect, self.host, self.port, self.client_id
        )
        
        # Start message loop
        self.msg_task = asyncio.create_task(
            asyncio.get_event_loop().run_in_executor(None, self.run)
        )
        
        # Wait for connection
        for i in range(10):
            if self.connected:
                return True
            await asyncio.sleep(1)
        
        return False
    
    async def check_symbol_availability(self, symbol: str, asset_class: str) -> Dict:
        """Check if a symbol is available and get its details"""
        contract = self._create_contract(symbol, asset_class)
        
        # Reset state
        self.current_request = symbol
        self.request_complete = False
        
        # Request contract details
        req_id = hash(symbol) % 10000
        
        await asyncio.get_event_loop().run_in_executor(
            None, self.reqContractDetails, req_id, contract
        )
        
        # Wait for response
        for i in range(10):  # 10 second timeout
            if self.request_complete:
                break
            await asyncio.sleep(1)
        
        # Return result
        if symbol in self.contract_details:
            return self.contract_details[symbol]
        else:
            return {
                'symbol': symbol,
                'available': False,
                'error': 'Contract not found'
            }
    
    def _create_contract(self, symbol: str, asset_class: str) -> Contract:
        """Create contract based on asset class"""
        contract = Contract()
        
        if asset_class == 'stocks':
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
        
        elif asset_class == 'forex':
            if '.' in symbol:
                base, quote = symbol.split('.')
            else:
                base, quote = symbol[:3], symbol[3:]
            contract.symbol = base
            contract.secType = "CASH"
            contract.currency = quote
            contract.exchange = "IDEALPRO"
        
        elif asset_class == 'futures':
            contract.symbol = symbol
            contract.secType = "FUT"
            contract.exchange = self._get_futures_exchange(symbol)
            contract.currency = "USD"
            # Add contract month
            contract.lastTradeDateOrContractMonth = self._get_front_month()
        
        elif asset_class == 'commodities':
            contract.symbol = symbol
            contract.secType = "FUT"
            contract.exchange = self._get_commodity_exchange(symbol)
            contract.currency = "USD"
            contract.lastTradeDateOrContractMonth = self._get_front_month()
        
        elif asset_class == 'crypto':
            contract.symbol = symbol
            contract.secType = "CRYPTO"
            contract.exchange = "PAXOS"
            contract.currency = "USD"
        
        return contract
    
    def _get_futures_exchange(self, symbol: str) -> str:
        """Get futures exchange"""
        exchanges = {
            'ES': 'CME', 'NQ': 'CME', 'RTY': 'CME', 'YM': 'CBOT',
            'ZB': 'CBOT', 'ZN': 'CBOT', 'ZF': 'CBOT', 'ZT': 'CBOT'
        }
        return exchanges.get(symbol, 'CME')
    
    def _get_commodity_exchange(self, symbol: str) -> str:
        """Get commodity exchange"""
        exchanges = {
            'GC': 'COMEX', 'SI': 'COMEX', 'HG': 'COMEX',
            'CL': 'NYMEX', 'NG': 'NYMEX', 'HO': 'NYMEX',
            'ZC': 'CBOT', 'ZS': 'CBOT', 'ZW': 'CBOT'
        }
        return exchanges.get(symbol, 'NYMEX')
    
    def _get_front_month(self) -> str:
        """Get front month contract"""
        now = datetime.now()
        year = now.year
        month = now.month + 1
        if month > 12:
            month = 1
            year += 1
        return f"{year}{month:02d}"
    
    async def discover_symbols(self, symbol_lists: Dict[str, List[str]]) -> Dict:
        """Discover available symbols across asset classes"""
        results = {}
        
        for asset_class, symbols in symbol_lists.items():
            print(f"\n🔍 Checking {asset_class.upper()} ({len(symbols)} symbols)")
            results[asset_class] = {}
            
            for symbol in symbols:
                try:
                    result = await self.check_symbol_availability(symbol, asset_class)
                    results[asset_class][symbol] = result
                    await asyncio.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print(f"❌ {symbol}: Error - {e}")
                    results[asset_class][symbol] = {'available': False, 'error': str(e)}
        
        return results
    
    async def disconnect(self):
        """Disconnect from IBKR"""
        if hasattr(self, 'msg_task'):
            self.msg_task.cancel()
        await asyncio.get_event_loop().run_in_executor(None, super().disconnect)

# Comprehensive symbol lists
SYMBOL_UNIVERSE = {
    'stocks': [
        # Large Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
        # Large Cap Traditional
        'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'WMT',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'GLD', 'SLV', 'TLT', 'HYG',
        # Additional popular stocks
        'KO', 'PEP', 'NKE', 'ADBE', 'CRM', 'INTC', 'AMD', 'MU', 'PYPL'
    ],
    
    'forex': [
        # Major pairs
        'EUR.USD', 'GBP.USD', 'USD.JPY', 'USD.CHF', 'AUD.USD', 'USD.CAD', 'NZD.USD',
        # Cross pairs
        'EUR.GBP', 'EUR.JPY', 'GBP.JPY', 'AUD.JPY', 'EUR.CHF', 'GBP.CHF',
        # Alternative formats
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD'
    ],
    
    'futures': [
        # Index futures
        'ES', 'NQ', 'RTY', 'YM',
        # Bond futures
        'ZB', 'ZN', 'ZF', 'ZT',
        # Currency futures
        '6E', '6B', '6J', '6S'
    ],
    
    'commodities': [
        # Metals
        'GC', 'SI', 'HG', 'PL',
        # Energy
        'CL', 'NG', 'HO', 'RB',
        # Agriculture
        'ZC', 'ZS', 'ZW', 'CT', 'SB', 'CC', 'KC'
    ],
    
    'crypto': [
        'BTC', 'ETH', 'LTC', 'BCH'
    ]
}

async def main():
    """Main symbol discovery function"""
    print("🔍 IBKR Symbol Discovery Tool")
    print("=" * 60)
    
    discovery = SymbolDiscovery()
    
    try:
        # Connect
        print("🔌 Connecting to IBKR...")
        if not await discovery.connect_to_ibkr():
            print("❌ Could not connect to IBKR")
            return
        
        # Discover symbols
        print("🔍 Starting symbol discovery...")
        results = await discovery.discover_symbols(SYMBOL_UNIVERSE)
        
        # Generate report
        generate_symbol_report(results)
        
        # Save results
        save_results(results)
        
    except Exception as e:
        print(f"❌ Discovery failed: {e}")
    finally:
        await discovery.disconnect()

def generate_symbol_report(results: Dict):
    """Generate human-readable report"""
    print("\n📊 SYMBOL DISCOVERY REPORT")
    print("=" * 60)
    
    total_symbols = 0
    available_symbols = 0
    
    for asset_class, symbols in results.items():
        class_available = sum(1 for s in symbols.values() if s.get('available', False))
        class_total = len(symbols)
        
        total_symbols += class_total
        available_symbols += class_available
        
        print(f"\n📈 {asset_class.upper()}: {class_available}/{class_total} available")
        
        # Show available symbols
        available_list = [symbol for symbol, data in symbols.items() 
                         if data.get('available', False)]
        if available_list:
            print(f"   ✅ Available: {', '.join(available_list[:10])}")
            if len(available_list) > 10:
                print(f"   ... and {len(available_list) - 10} more")
        
        # Show unavailable symbols
        unavailable_list = [symbol for symbol, data in symbols.items() 
                           if not data.get('available', False)]
        if unavailable_list:
            print(f"   ❌ Not available: {', '.join(unavailable_list[:5])}")
            if len(unavailable_list) > 5:
                print(f"   ... and {len(unavailable_list) - 5} more")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total symbols tested: {total_symbols}")
    print(f"   Available symbols: {available_symbols}")
    print(f"   Success rate: {available_symbols/total_symbols*100:.1f}%")

def save_results(results: Dict):
    """Save results to files"""
    # Save detailed JSON
    with open('symbol_discovery_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save CSV for each asset class
    for asset_class, symbols in results.items():
        filename = f'available_symbols_{asset_class}.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Symbol', 'Available', 'Exchange', 'Currency', 'Long_Name'])
            
            for symbol, data in symbols.items():
                if data.get('available', False):
                    writer.writerow([
                        symbol,
                        'Yes',
                        data.get('exchange', ''),
                        data.get('currency', ''),
                        data.get('longName', '')
                    ])
    
    # Create master available symbols list
    all_available = []
    for asset_class, symbols in results.items():
        for symbol, data in symbols.items():
            if data.get('available', False):
                all_available.append({
                    'symbol': symbol,
                    'asset_class': asset_class,
                    'exchange': data.get('exchange', ''),
                    'currency': data.get('currency', ''),
                    'long_name': data.get('longName', '')
                })
    
    with open('all_available_symbols.csv', 'w', newline='') as f:
        if all_available:
            writer = csv.DictWriter(f, fieldnames=all_available[0].keys())
            writer.writeheader()
            writer.writerows(all_available)
    
    print(f"\n💾 Results saved:")
    print(f"   📄 symbol_discovery_results.json (detailed results)")
    print(f"   📄 all_available_symbols.csv (master list)")
    print(f"   📄 available_symbols_*.csv (by asset class)")

if __name__ == "__main__":
    try:
        import logging
        logging.basicConfig(level=logging.WARNING)
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Discovery cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")