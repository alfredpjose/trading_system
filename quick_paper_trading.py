#!/usr/bin/env python3
"""
IBKR Crypto Paper Trading Test - 24/7 Compatible
Tests paper trading with Bitcoin (BTC/USD) 

IBKR Crypto Details:
- Available: BTC, ETH, LTC, BCH
- Exchange: PAXOS
- Trading: 24/7
- Minimum: 0.00001 BTC

Usage: python crypto_paper_test.py
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

class CryptoPaperTest(EWrapper, EClient):
    """IBKR Crypto paper trading test"""
    
    def __init__(self):
        EClient.__init__(self, self)
        
        # Configuration
        self.host = os.getenv('IBKR_HOST', '127.0.0.1')
        self.port = int(os.getenv('IBKR_PORT', '7497'))
        self.client_id = int(os.getenv('IBKR_CLIENT_ID', '1000')) + 777
        
        # Crypto test settings
        self.test_symbol = "BTC"  # Bitcoin
        self.test_currency = "USD"
        self.test_quantity = 0.001  # Small amount for testing (~$30-60)
        
        # State
        self.connected = False
        self.next_order_id = None
        self.orders = {}
        self.positions = {}
        self.account_info = {}
        self.market_data_received = False
        self.current_price = None
        self.contract_details_received = False
    
    def connectAck(self):
        self.connected = True
        print("✅ Connected to IBKR")
    
    def nextValidId(self, orderId):
        self.next_order_id = orderId
        print(f"📝 Ready to trade Crypto (Order ID: {orderId})")
    
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in [2104, 2106, 2158, 2119, 2137]:  # Info messages
            print(f"ℹ️  {errorString}")
        elif errorCode in [10089, 10090]:  # Market data warnings
            print(f"📊 Market Data: {errorString}")
        elif errorCode == 200:  # No security definition found
            print(f"❌ Contract Error: {errorString}")
            print("   This usually means crypto is not available in your paper account")
        elif errorCode == 201:  # Order rejected
            print(f"❌ Order Rejected: {errorString}")
        else:
            print(f"⚠️  Error {errorCode}: {errorString}")
    
    def contractDetails(self, reqId, contractDetails):
        """Handle contract details response"""
        self.contract_details_received = True
        contract = contractDetails.contract
        print(f"✅ Contract found: {contract.symbol}/{contract.currency} on {contract.exchange}")
        print(f"   Min tick: {contractDetails.minTick}")
        print(f"   Contract ID: {contract.conId}")
    
    def contractDetailsEnd(self, reqId):
        """End of contract details"""
        if not self.contract_details_received:
            print("❌ No contract details received - crypto may not be available")
    
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        action = self.orders.get(orderId, {}).get('action', 'UNKNOWN')
        print(f"📊 Order {orderId} ({action}): {status}")
        if filled > 0:
            print(f"   💰 Filled {filled} of {filled + remaining} BTC at ${avgFillPrice:,.2f}")
        if status == "Filled":
            self.orders[orderId]['filled'] = True
            self.orders[orderId]['fill_price'] = avgFillPrice
        elif "Cancelled" in status:
            self.orders[orderId]['cancelled'] = True
    
    def position(self, account, contract, position, avgCost):
        if contract.secType == "CRYPTO" and contract.symbol == self.test_symbol:
            self.positions[f"{contract.symbol}.{contract.currency}"] = {
                'position': position,
                'avg_cost': avgCost,
                'contract': contract,
                'market_value': position * avgCost if avgCost > 0 else 0
            }
            print(f"📈 Crypto Position: {contract.symbol}/{contract.currency} = {position:.6f} BTC @ ${avgCost:,.2f}")
    
    def tickPrice(self, reqId, tickType, price, attrib):
        """Handle real-time crypto price updates"""
        if tickType == 4:  # Last price
            self.current_price = price
            self.market_data_received = True
            print(f"₿ BTC/USD Price: ${price:,.2f}")
    
    def accountSummary(self, reqId, account, tag, value, currency):
        """Handle account summary"""
        if tag in ['TotalCashValue', 'NetLiquidation', 'BuyingPower']:
            self.account_info[tag] = float(value)
            print(f"💰 {tag}: ${float(value):,.2f}")
    
    async def connect_and_setup(self):
        """Connect and setup"""
        print(f"🔌 Connecting to IBKR at {self.host}:{self.port}...")
        
        # Connect
        await asyncio.get_event_loop().run_in_executor(
            None, self.connect, self.host, self.port, self.client_id
        )
        
        # Start message loop
        self.msg_task = asyncio.create_task(
            asyncio.get_event_loop().run_in_executor(None, self.run)
        )
        
        # Wait for connection
        for i in range(15):
            if self.connected and self.next_order_id:
                return True
            await asyncio.sleep(1)
        
        return False
    
    def create_crypto_contract(self):
        """Create BTC/USD crypto contract"""
        contract = Contract()
        contract.symbol = self.test_symbol
        contract.secType = "CRYPTO"
        contract.currency = self.test_currency
        contract.exchange = "PAXOS"  # IBKR's crypto exchange
        return contract
    
    async def check_crypto_availability(self):
        """Check if crypto contract is available"""
        print("🔍 Checking if crypto is available in your account...")
        contract = self.create_crypto_contract()
        
        await asyncio.get_event_loop().run_in_executor(
            None, self.reqContractDetails, 1, contract
        )
        
        # Wait for contract details
        for i in range(10):
            if self.contract_details_received:
                return True
            await asyncio.sleep(1)
        
        return False
    
    def create_crypto_order(self, action, quantity):
        """Create crypto market order"""
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity
        return order
    
    async def get_crypto_market_data(self):
        """Subscribe to BTC/USD market data"""
        print("📊 Subscribing to BTC/USD market data...")
        contract = self.create_crypto_contract()
        
        await asyncio.get_event_loop().run_in_executor(
            None, self.reqMktData, 2, contract, "", False, False, []
        )
        
        # Wait for market data
        for i in range(15):  # Longer wait for crypto
            if self.market_data_received:
                break
            await asyncio.sleep(1)
        
        return self.market_data_received
    
    async def place_crypto_order(self, action):
        """Place crypto order"""
        contract = self.create_crypto_contract()
        order = self.create_crypto_order(action, self.test_quantity)
        
        order_id = self.next_order_id
        self.next_order_id += 1
        
        # Track order
        self.orders[order_id] = {
            'action': action,
            'quantity': self.test_quantity,
            'filled': False,
            'cancelled': False,
            'timestamp': datetime.now()
        }
        
        estimated_value = self.test_quantity * (self.current_price or 50000)
        print(f"📝 Placing {action} order for {self.test_quantity} BTC (~${estimated_value:.2f})...")
        
        # Place order
        await asyncio.get_event_loop().run_in_executor(
            None, self.placeOrder, order_id, contract, order
        )
        
        print(f"✅ {action} order {order_id} submitted")
        return order_id
    
    async def get_positions(self):
        """Request current positions"""
        await asyncio.get_event_loop().run_in_executor(None, self.reqPositions)
        await asyncio.sleep(3)
    
    async def get_account_summary(self):
        """Get account summary"""
        await asyncio.get_event_loop().run_in_executor(
            None, self.reqAccountSummary, 1, "All", "TotalCashValue,NetLiquidation,BuyingPower"
        )
        await asyncio.sleep(2)
    
    async def run_test(self):
        """Run the complete crypto test"""
        print("₿ IBKR Crypto Paper Trading Test (BTC/USD)")
        print("=" * 60)
        
        try:
            # Step 1: Connect
            if not await self.connect_and_setup():
                print("❌ Connection failed")
                return False
            
            # Step 2: Check crypto availability
            print("\n🔍 Checking crypto availability...")
            crypto_available = await self.check_crypto_availability()
            
            if not crypto_available:
                print("❌ Crypto trading not available in your paper account")
                print("\n💡 To enable crypto in IBKR paper trading:")
                print("   1. Log into IBKR Client Portal")
                print("   2. Go to Settings → Account Settings")
                print("   3. Trading Permissions → Cryptocurrencies")
                print("   4. Enable crypto trading for paper account")
                print("   5. Note: May take time to activate")
                return False
            
            # Step 3: Get account info
            print("\n💰 Getting account information...")
            await self.get_account_summary()
            
            # Step 4: Get crypto market data
            print(f"\n₿ Getting BTC/USD market data...")
            data_received = await self.get_crypto_market_data()
            
            if data_received:
                print(f"✅ Current BTC price: ${self.current_price:,.2f}")
                estimated_order_value = self.test_quantity * self.current_price
                print(f"📊 Test order value: ~${estimated_order_value:.2f}")
            else:
                print("⚠️ No market data received - orders may still work")
            
            # Step 5: Get initial positions
            print("\n📈 Getting initial crypto positions...")
            await self.get_positions()
            initial_btc_pos = self.positions.get("BTC.USD", {}).get('position', 0)
            print(f"Initial BTC position: {initial_btc_pos:.6f}")
            
            # Step 6: Place BUY order
            print(f"\n🛒 Step 1: Buying {self.test_quantity} BTC")
            buy_order_id = await self.place_crypto_order("BUY")
            
            # Wait for fill
            print("⏳ Waiting for BUY order to fill...")
            for i in range(45):  # Longer timeout for crypto
                order_info = self.orders.get(buy_order_id, {})
                if order_info.get('filled', False):
                    break
                if order_info.get('cancelled', False):
                    print("❌ BUY order was cancelled")
                    break
                await asyncio.sleep(1)
                if i % 10 == 0 and i > 0:  # Progress every 10 seconds
                    print(f"   ... still waiting ({i}s)")
            
            # Check buy status
            buy_filled = self.orders.get(buy_order_id, {}).get('filled', False)
            if buy_filled:
                fill_price = self.orders[buy_order_id].get('fill_price', 0)
                print(f"✅ BUY order filled at ${fill_price:,.2f}")
            else:
                print("⚠️ BUY order not filled yet")
            
            # Check position after buy
            await self.get_positions()
            after_buy_pos = self.positions.get("BTC.USD", {}).get('position', 0)
            
            # Step 7: Place SELL order (only if we have position)
            if after_buy_pos > 0 or buy_filled:
                print(f"\n💰 Step 2: Selling {self.test_quantity} BTC")
                sell_order_id = await self.place_crypto_order("SELL")
                
                # Wait for fill
                print("⏳ Waiting for SELL order to fill...")
                for i in range(45):
                    order_info = self.orders.get(sell_order_id, {})
                    if order_info.get('filled', False):
                        break
                    if order_info.get('cancelled', False):
                        print("❌ SELL order was cancelled")
                        break
                    await asyncio.sleep(1)
                    if i % 10 == 0 and i > 0:
                        print(f"   ... still waiting ({i}s)")
                
                # Check sell status
                sell_filled = self.orders.get(sell_order_id, {}).get('filled', False)
                if sell_filled:
                    fill_price = self.orders[sell_order_id].get('fill_price', 0)
                    print(f"✅ SELL order filled at ${fill_price:,.2f}")
                else:
                    print("⚠️ SELL order not filled yet")
            else:
                print("\n⚠️ Skipping SELL order - no BTC position to sell")
                sell_filled = False
            
            # Final position check
            await self.get_positions()
            final_btc_pos = self.positions.get("BTC.USD", {}).get('position', 0)
            
            # Results
            print(f"\n📊 Crypto Test Summary:")
            print(f"Initial BTC position: {initial_btc_pos:.6f}")
            print(f"After BUY: {after_buy_pos:.6f}")
            print(f"Final BTC position: {final_btc_pos:.6f}")
            
            print(f"\n🎯 Order Results:")
            print(f"BUY order filled: {'✅' if buy_filled else '❌'}")
            if 'sell_filled' in locals():
                print(f"SELL order filled: {'✅' if sell_filled else '❌'}")
            
            if buy_filled and (not 'sell_filled' in locals() or sell_filled):
                print(f"\n🎉 SUCCESS! Crypto paper trading is working!")
                print("Bitcoin orders executed successfully.")
                
                # Calculate profit/loss if both orders filled
                if 'sell_filled' in locals() and sell_filled and buy_filled:
                    buy_price = self.orders[buy_order_id].get('fill_price', 0)
                    sell_price = self.orders[sell_order_id].get('fill_price', 0)
                    if buy_price and sell_price:
                        pnl = (sell_price - buy_price) * self.test_quantity
                        print(f"📊 Round-trip P&L: ${pnl:.2f}")
                
            elif buy_filled:
                print(f"\n⚠️ PARTIAL SUCCESS")
                print("BUY order worked - crypto trading is available!")
                
            else:
                print(f"\n❌ No orders filled")
                print("Possible reasons:")
                print("- Crypto permissions not enabled")
                print("- Insufficient buying power")
                print("- Market connectivity issues")
                print("- Order size too small")
            
            print(f"\n₿ Crypto Trading Notes:")
            print("- IBKR offers: BTC, ETH, LTC, BCH")
            print("- Trading is 24/7 (even weekends)")
            print("- Minimum order: 0.00001 BTC")
            print("- Exchange: PAXOS")
            print("- Paper account needs crypto permissions")
            
            return True
            
        except Exception as e:
            print(f"❌ Crypto test failed: {e}")
            return False
        
        finally:
            # Cleanup
            try:
                if hasattr(self, 'msg_task'):
                    self.msg_task.cancel()
                await asyncio.get_event_loop().run_in_executor(None, self.disconnect)
                print("\n🔌 Disconnected from IBKR")
            except:
                pass

async def main():
    """Main function"""
    print("₿ IBKR Crypto Paper Trading Test")
    print("=" * 70)
    
    # Environment info
    print(f"📍 IBKR Host: {os.getenv('IBKR_HOST', '127.0.0.1')}")
    print(f"📍 IBKR Port: {os.getenv('IBKR_PORT', '7497')} (Paper Trading)")
    print(f"📍 Cryptocurrency: BTC/USD")
    print(f"📍 Test Size: 0.001 BTC (~$30-60)")
    print(f"📍 Exchange: PAXOS")
    
    # Market info
    now = datetime.now()
    print(f"📅 Current Time: {now.strftime('%A %Y-%m-%d %H:%M')}")
    print(f"🕐 Crypto Market: OPEN (24/7)")
    
    # IBKR Crypto info
    print(f"\n₿ IBKR Crypto Details:")
    print("   Available: BTC, ETH, LTC, BCH")
    print("   Trading Hours: 24/7")
    print("   Min Order: 0.00001 BTC")
    print("   Settlement: T+0 (immediate)")
    
    # Prerequisites
    print(f"\n✅ Prerequisites:")
    print("   - IBKR TWS or IB Gateway is running")
    print("   - Paper Trading mode is enabled")
    print("   - Crypto permissions enabled in paper account")
    print("   - Sufficient USD buying power")
    print("   - API connections are enabled in TWS")
    
    # Quick connection test
    print(f"\n🔌 Testing connection to IBKR...")
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((os.getenv('IBKR_HOST', '127.0.0.1'), 
                                int(os.getenv('IBKR_PORT', '7497'))))
        sock.close()
        
        if result == 0:
            print("✅ IBKR is reachable")
        else:
            print("❌ Cannot reach IBKR - make sure TWS/Gateway is running")
            return
    except:
        print("❌ Connection test failed")
        return
    
    # Warning about crypto permissions
    print(f"\n⚠️  Important Notes:")
    print("   - Crypto may not be enabled by default in paper accounts")
    print("   - You may need to request crypto permissions from IBKR")
    print("   - This test will detect if crypto is available")
    
    # Run test
    print(f"\nStarting crypto test in 3 seconds...")
    await asyncio.sleep(3)
    
    test = CryptoPaperTest()
    await test.run_test()
    
    print(f"\n📝 Next Steps:")
    print("   1. Check TWS 'Portfolio' for crypto positions")
    print("   2. If permissions error: Enable crypto in IBKR settings")
    print("   3. Try other cryptos: ETH, LTC, BCH")
    print("   4. Crypto trades 24/7 - perfect for weekend testing!")

if __name__ == "__main__":
    try:
        # Simple logging setup
        import logging
        logging.basicConfig(level=logging.WARNING)
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nCrypto Trading Troubleshooting:")
        print("1. Enable crypto permissions in IBKR Client Portal")
        print("2. Check paper account has crypto enabled")
        print("3. Verify sufficient USD buying power")
        print("4. Try smaller position size (0.0001 BTC)")
        print("5. Contact IBKR support for crypto access")