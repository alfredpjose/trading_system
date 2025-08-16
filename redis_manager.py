#!/usr/bin/env python3
"""
Redis Auto-Initialization for Trading System
Automatically starts Redis Docker container when needed

Add this to your trading system startup
"""

import subprocess
import time
import socket
import os
import sys
from pathlib import Path
from loguru import logger

class RedisManager:
    """Manages Redis Docker container for trading system"""
    
    def __init__(self):
        self.container_name = "trading-redis"
        self.redis_port = 6379
        self.redis_password = os.getenv('REDIS_PASSWORD', '')
        self.docker_image = "redis:7-alpine"
        
    def is_redis_running(self) -> bool:
        """Check if Redis is accessible"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', self.redis_port))
            sock.close()
            return result == 0
        except:
            return False
    
    def is_docker_available(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def is_container_running(self) -> bool:
        """Check if Redis container is running"""
        try:
            result = subprocess.run([
                'docker', 'ps', '--filter', f'name={self.container_name}', 
                '--format', '{{.Names}}'
            ], capture_output=True, text=True)
            
            return self.container_name in result.stdout
        except:
            return False
    
    def start_redis_container(self) -> bool:
        """Start Redis Docker container"""
        try:
            logger.info("🐳 Starting Redis Docker container...")
            
            # Check if container exists but is stopped
            result = subprocess.run([
                'docker', 'ps', '-a', '--filter', f'name={self.container_name}',
                '--format', '{{.Names}}'
            ], capture_output=True, text=True)
            
            if self.container_name in result.stdout:
                # Container exists, just start it
                logger.info("📦 Starting existing Redis container...")
                subprocess.run(['docker', 'start', self.container_name], check=True)
            else:
                # Create new container
                logger.info("🆕 Creating new Redis container...")
                cmd = [
                    'docker', 'run', '-d',
                    '--name', self.container_name,
                    '-p', f'{self.redis_port}:6379',
                    '--restart', 'unless-stopped'
                ]
                
                # Add password if specified
                if self.redis_password:
                    cmd.extend(['--env', f'REDIS_PASSWORD={self.redis_password}'])
                    cmd.extend([self.docker_image, 'redis-server', '--requirepass', self.redis_password])
                else:
                    cmd.append(self.docker_image)
                
                subprocess.run(cmd, check=True)
            
            # Wait for Redis to be ready
            logger.info("⏳ Waiting for Redis to be ready...")
            for i in range(30):  # 30 second timeout
                if self.is_redis_running():
                    logger.success(f"✅ Redis is ready after {i+1} seconds")
                    return True
                time.sleep(1)
            
            logger.error("❌ Redis failed to start within 30 seconds")
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Docker command failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to start Redis: {e}")
            return False
    
    def stop_redis_container(self):
        """Stop Redis container"""
        try:
            logger.info("🛑 Stopping Redis container...")
            subprocess.run(['docker', 'stop', self.container_name], check=True)
            logger.info("✅ Redis container stopped")
        except:
            logger.warning("⚠️ Could not stop Redis container")
    
    def ensure_redis_running(self) -> bool:
        """Ensure Redis is running, start if needed"""
        # Check if Redis is already accessible
        if self.is_redis_running():
            logger.info("✅ Redis is already running")
            return True
        
        # Check if Docker is available
        if not self.is_docker_available():
            logger.error("❌ Docker is not available")
            logger.info("💡 Please install Docker or start Redis manually")
            return False
        
        # Start Redis container
        return self.start_redis_container()
    
    def test_redis_connection(self) -> bool:
        """Test Redis connection with Python redis client"""
        try:
            import redis
            
            # Create Redis client
            if self.redis_password:
                r = redis.Redis(host='localhost', port=self.redis_port, 
                              password=self.redis_password, decode_responses=True)
            else:
                r = redis.Redis(host='localhost', port=self.redis_port, 
                              decode_responses=True)
            
            # Test ping
            r.ping()
            
            # Test basic operations
            r.set('trading_system_test', 'working')
            result = r.get('trading_system_test')
            r.delete('trading_system_test')
            
            if result == 'working':
                logger.success("✅ Redis connection test successful")
                return True
            else:
                logger.error("❌ Redis test failed")
                return False
                
        except ImportError:
            logger.warning("⚠️ Redis Python client not installed")
            logger.info("📦 Install with: pip install redis")
            return False
        except Exception as e:
            logger.error(f"❌ Redis connection test failed: {e}")
            return False

# Standalone functions for easy integration
def ensure_redis_for_trading():
    """Ensure Redis is running for trading system"""
    redis_manager = RedisManager()
    return redis_manager.ensure_redis_running()

def test_redis_setup():
    """Test complete Redis setup"""
    redis_manager = RedisManager()
    
    print("🧪 Testing Redis Setup")
    print("=" * 40)
    
    # Test Docker availability
    if redis_manager.is_docker_available():
        print("✅ Docker is available")
    else:
        print("❌ Docker is not available")
        return False
    
    # Ensure Redis is running
    if redis_manager.ensure_redis_running():
        print("✅ Redis container is running")
    else:
        print("❌ Failed to start Redis")
        return False
    
    # Test Redis connection
    if redis_manager.test_redis_connection():
        print("✅ Redis connection working")
        print("🎉 Redis setup is complete!")
        return True
    else:
        print("❌ Redis connection failed")
        return False

# Integration with your main.py
def add_redis_to_main_py():
    """Example integration with main.py"""
    
    # Add this to the beginning of your TradingSystem.__init__ or main() function:
    
    example_code = """
    # Add to your main.py imports:
    from redis_manager import ensure_redis_for_trading
    
    # Add to TradingSystem.__init__ or main() function:
    async def initialize(self):
        logger.info("Initializing trading system...")
        
        # Ensure Redis is running
        if not ensure_redis_for_trading():
            logger.error("Failed to start Redis - continuing without cache")
            self.cache = None  # Fallback to no cache
        else:
            self.cache = DataCache(self.redis_url, self.cache_ttl)
            await self.cache.connect()
        
        # Rest of your initialization...
    """
    
    return example_code

# WSL2 specific functions
def setup_wsl2_redis():
    """Setup Redis for WSL2 environment"""
    print("🐧 WSL2 Redis Setup")
    print("=" * 30)
    
    # Check if running in WSL2
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read()
            if 'microsoft' in version_info.lower():
                print("✅ Running in WSL2")
                
                # WSL2 specific Docker commands
                wsl_commands = [
                    "# Start Docker service in WSL2:",
                    "sudo service docker start",
                    "",
                    "# Or if using Docker Desktop:",
                    "# Make sure Docker Desktop is running on Windows",
                    "",
                    "# Test Docker in WSL2:",
                    "docker --version",
                    "",
                    "# Redis will be accessible on localhost:6379"
                ]
                
                print("💡 WSL2 Docker Setup:")
                for cmd in wsl_commands:
                    print(f"   {cmd}")
                
                return True
            else:
                print("ℹ️ Not running in WSL2")
                return False
    except:
        print("ℹ️ Could not detect WSL2")
        return False

if __name__ == "__main__":
    """Run Redis setup test"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "wsl2":
        setup_wsl2_redis()
    else:
        success = test_redis_setup()
        
        if success:
            print("\n🎯 Next Steps:")
            print("1. Add Redis manager to your trading system")
            print("2. Update your .env file:")
            print("   DATA_REDIS_URL=redis://localhost:6379")
            print("3. Redis will auto-start when needed")
        else:
            print("\n❌ Redis setup failed")
            print("Check Docker installation and try again")