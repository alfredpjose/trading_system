#!/usr/bin/env python3
"""
IB Gateway Debug Test
Debug specific connection issues with IB Gateway

Usage: python gateway_debug_test.py
"""

import asyncio
import os
import sys
import time
import socket
import threading
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class GatewayDebugTest(EWrapper, EClient):
    """Debug test for IB Gateway"""
    
    def __init__(self):
        EClient.__init__(self, self)
        
        # Configuration - try multiple client IDs
        self.host = os.getenv('IBKR_HOST', '127.0.0.1')
        self.port = int(os.getenv('IBKR_PORT', '7497'))
        self.base_client_id = int(os.getenv('IBKR_CLIENT_ID', '1000'))
        
        # State
        self.connected = False
        self.ready = False
        self.connection_attempts = 0
        self.messages = []
        self.start_time = None
        
    def connectAck(self):
        self.connected = True
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"✅ CONNECTION ACK received after {elapsed:.1f}s")
        
    def nextValidId(self, orderId):
        self.ready = True
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"✅ READY signal received after {elapsed:.1f}s (Order ID: {orderId})")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        self.messages.append({
            'reqId': reqId,
            'errorCode': errorCode, 
            'errorString': errorString,
            'timestamp': time.time()
        })
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        if errorCode == -1:
            print(f"🔗 [{elapsed:.1f}s] System: {errorString}")
        elif errorCode in [2104, 2106, 2158, 2119, 2137]:
            print(f"ℹ️  [{elapsed:.1f}s] Info {errorCode}: {errorString}")
        elif errorCode in [502, 503, 504]:
            print(f"❌ [{elapsed:.1f}s] Connection Error {errorCode}: {errorString}")
        elif errorCode == 507:
            print(f"⚠️  [{elapsed:.1f}s] Bad Message: {errorString}")
        else:
            print(f"⚠️  [{elapsed:.1f}s] Error {errorCode}: {errorString}")

def test_raw_socket_connection(host, port):
    """Test raw socket connection with detailed output"""
    print(f"\n🔌 Testing Raw Socket Connection to {host}:{port}")
    
    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        print(f"   📡 Attempting to connect...")
        start_time = time.time()
        
        result = sock.connect_ex((host, port))
        elapsed = time.time() - start_time
        
        if result == 0:
            print(f"   ✅ Socket connected successfully in {elapsed:.3f}s")
            
            # Try to send a simple message to see if it responds
            try:
                sock.send(b"test")
                print(f"   📤 Test message sent")
            except:
                print(f"   ⚠️  Could not send test message")
            
            sock.close()
            return True
        else:
            print(f"   ❌ Socket connection failed: Error {result}")
            return False
            
    except socket.timeout:
        print(f"   ⏰ Socket connection timed out after 10s")
        return False
    except Exception as e:
        print(f"   ❌ Socket error: {e}")
        return False

def check_port_conflicts():
    """Check for port conflicts"""
    print(f"\n🔍 Checking for Port Conflicts")
    
    common_ports = [7496, 7497, 4001, 4002]
    
    for port in common_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result == 0:
                status = "🟢 OPEN"
                if port == int(os.getenv('IBKR_PORT', '7497')):
                    status += " (YOUR PORT)"
            else:
                status = "🔴 CLOSED"
                
            print(f"   Port {port}: {status}")
            
        except:
            print(f"   Port {port}: ❌ ERROR")

async def test_connection_with_timeout(host, port, client_id, timeout_seconds=20):
    """Test connection with detailed timeout handling"""
    print(f"\n🧪 Testing IBKR API Connection")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Client ID: {client_id}")
    print(f"   Timeout: {timeout_seconds}s")
    
    test = GatewayDebugTest()
    test.start_time = time.time()
    
    try:
        print(f"\n⏳ Connecting...")
        
        # Connect with executor
        connect_future = asyncio.get_event_loop().run_in_executor(
            None, test.connect, host, port, client_id
        )
        
        # Wait for connection attempt to complete
        try:
            await asyncio.wait_for(connect_future, timeout=5.0)
            print(f"   📡 Connection attempt completed")
        except asyncio.TimeoutError:
            print(f"   ⏰ Connection attempt timed out after 5s")
            return False
        
        # Start message loop
        msg_task = asyncio.create_task(
            asyncio.get_event_loop().run_in_executor(None, test.run)
        )
        
        # Wait for connection with progress updates
        print(f"\n⏳ Waiting for connection acknowledgment...")
        
        for i in range(timeout_seconds):
            elapsed = time.time() - test.start_time
            
            if test.connected and test.ready:
                print(f"\n🎉 SUCCESS! Fully connected in {elapsed:.1f}s")
                break
            elif test.connected:
                print(f"   🔗 Connected, waiting for ready signal... ({elapsed:.1f}s)")
            else:
                if i % 5 == 0 and i > 0:
                    print(f"   ⏳ Still waiting for connection... ({elapsed:.1f}s)")
                    
            await asyncio.sleep(1)
        
        # Final status
        if not test.connected:
            elapsed = time.time() - test.start_time
            print(f"\n❌ TIMEOUT: No connection after {elapsed:.1f}s")
            print(f"   This usually means:")
            print(f"   - API not enabled in Gateway")
            print(f"   - Wrong port number")
            print(f"   - Client ID conflict")
            print(f"   - Gateway not running")
        elif not test.ready:
            elapsed = time.time() - test.start_time
            print(f"\n⚠️  PARTIAL: Connected but not ready after {elapsed:.1f}s")
            print(f"   Connection acknowledged but no order ID received")
        
        # Show messages received
        if test.messages:
            print(f"\n📋 Messages Received:")
            for msg in test.messages:
                elapsed = msg['timestamp'] - test.start_time
                print(f"   [{elapsed:.1f}s] {msg['errorCode']}: {msg['errorString']}")
        
        # Cleanup
        msg_task.cancel()
        try:
            await msg_task
        except asyncio.CancelledError:
            pass
        
        await asyncio.get_event_loop().run_in_executor(None, test.disconnect)
        
        return test.connected and test.ready
        
    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        return False

async def try_different_client_ids(host, port, base_id):
    """Try connecting with different client IDs"""
    print(f"\n🔄 Trying Different Client IDs")
    
    client_ids_to_try = [
        base_id,
        base_id + 1, 
        base_id + 10,
        base_id + 100,
        1,
        999,
        1234,
        5678
    ]
    
    for client_id in client_ids_to_try:
        print(f"\n   Trying Client ID: {client_id}")
        
        # Quick test with 10 second timeout
        success = await test_connection_with_timeout(host, port, client_id, 10)
        
        if success:
            print(f"✅ SUCCESS with Client ID {client_id}!")
            print(f"💡 Update your .env file: IBKR_CLIENT_ID={client_id}")
            return client_id
        
        await asyncio.sleep(1)  # Brief pause between attempts
    
    print(f"❌ No client ID worked")
    return None

async def main():
    """Main debug function"""
    print("🔧 IB Gateway Debug Test")
    print("=" * 50)
    
    host = os.getenv('IBKR_HOST', '127.0.0.1')
    port = int(os.getenv('IBKR_PORT', '7497'))
    client_id = int(os.getenv('IBKR_CLIENT_ID', '1000'))
    
    print(f"📍 Current Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Client ID: {client_id}")
    
    # Step 1: Test raw socket
    socket_ok = test_raw_socket_connection(host, port)
    
    if not socket_ok:
        print(f"\n❌ Socket connection failed!")
        print(f"   Gateway is not responding on port {port}")
        print(f"   Check:")
        print(f"   1. Gateway is running and logged in")
        print(f"   2. Port {port} is correct for your Gateway")
        print(f"   3. API is enabled in Gateway settings")
        return
    
    # Step 2: Check port conflicts
    check_port_conflicts()
    
    # Step 3: Test IBKR API connection
    print(f"\n🧪 Testing IBKR API Connection...")
    success = await test_connection_with_timeout(host, port, client_id, 20)
    
    if success:
        print(f"\n🎉 Your Gateway connection is working perfectly!")
        print(f"   You can now run the paper trading tests.")
        return
    
    # Step 4: Try different client IDs
    working_client_id = await try_different_client_ids(host, port, client_id)
    
    if working_client_id:
        print(f"\n✅ Found working configuration!")
        print(f"   Client ID {working_client_id} works")
    else:
        print(f"\n❌ No configuration worked")
        print(f"\n🔧 Additional Troubleshooting:")
        print(f"   1. Restart IB Gateway completely")
        print(f"   2. Check Gateway API settings again")
        print(f"   3. Try port 4001 instead of {port}")
        print(f"   4. Disable Windows Firewall temporarily")
        print(f"   5. Check if another application is using the API")

if __name__ == "__main__":
    try:
        import logging
        logging.basicConfig(level=logging.CRITICAL)
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Debug test cancelled")
    except Exception as e:
        print(f"\n❌ Debug test failed: {e}")