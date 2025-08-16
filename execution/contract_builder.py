# execution/contract_builder.py
from ibapi.contract import Contract
from typing import Dict, Any
from datetime import datetime, timedelta

class UniversalContractBuilder:
    """Build IBKR contracts for all asset classes"""
    
    @staticmethod
    def build_contract(symbol: str, asset_class: str, **kwargs) -> Contract:
        """Build contract based on asset class"""
        contract = Contract()
        
        if asset_class == "stocks":
            return UniversalContractBuilder._build_stock_contract(symbol, **kwargs)
        elif asset_class == "forex":
            return UniversalContractBuilder._build_forex_contract(symbol, **kwargs)
        elif asset_class == "commodities":
            return UniversalContractBuilder._build_commodity_contract(symbol, **kwargs)
        elif asset_class == "futures":
            return UniversalContractBuilder._build_futures_contract(symbol, **kwargs)
        elif asset_class == "options":
            return UniversalContractBuilder._build_options_contract(symbol, **kwargs)
        elif asset_class == "crypto":
            return UniversalContractBuilder._build_crypto_contract(symbol, **kwargs)
        else:
            raise ValueError(f"Unsupported asset class: {asset_class}")
    
    @staticmethod
    def _build_stock_contract(symbol: str, exchange: str = "SMART", 
                             currency: str = "USD", **kwargs) -> Contract:
        """Build stock contract"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency
        
        # Handle international stocks
        if exchange in ["TSE", "LSE", "XETRA", "ASX", "HKEX"]:
            currency_map = {
                "TSE": "JPY", "LSE": "GBP", "XETRA": "EUR",
                "ASX": "AUD", "HKEX": "HKD"
            }
            contract.currency = currency_map.get(exchange, currency)
        
        return contract
    
    @staticmethod
    def _build_forex_contract(symbol: str, exchange: str = "IDEALPRO", **kwargs) -> Contract:
        """Build forex contract"""
        # Handle symbol formats: "EUR.USD" or "EURUSD"
        if "." in symbol:
            base, quote = symbol.split(".")
        else:
            base = symbol[:3]
            quote = symbol[3:]
        
        contract = Contract()
        contract.symbol = base
        contract.secType = "CASH"
        contract.currency = quote
        contract.exchange = exchange
        return contract
    
    @staticmethod
    def _build_commodity_contract(symbol: str, exchange: str = None, 
                                 currency: str = "USD", **kwargs) -> Contract:
        """Build commodity futures contract"""
        # Commodity exchange mapping
        exchange_map = {
            "GC": "COMEX",    # Gold
            "SI": "COMEX",    # Silver
            "HG": "COMEX",    # Copper
            "CL": "NYMEX",    # Crude Oil
            "NG": "NYMEX",    # Natural Gas
            "ZC": "CBOT",     # Corn
            "ZS": "CBOT",     # Soybeans
            "ZW": "CBOT",     # Wheat
        }
        
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "FUT"
        contract.exchange = exchange or exchange_map.get(symbol, "NYMEX")
        contract.currency = currency
        
        # Add contract month (next quarter)
        contract.lastTradeDateOrContractMonth = UniversalContractBuilder._get_next_contract_month()
        
        return contract
    
    @staticmethod
    def _build_futures_contract(symbol: str, exchange: str = None, 
                               currency: str = "USD", **kwargs) -> Contract:
        """Build futures contract"""
        # Futures exchange mapping
        exchange_map = {
            "ES": "CME",      # S&P 500 E-mini
            "NQ": "CME",      # NASDAQ E-mini
            "RTY": "CME",     # Russell 2000 E-mini
            "YM": "CBOT",     # Dow E-mini
            "ZB": "CBOT",     # 30-Year Treasury
            "ZN": "CBOT",     # 10-Year Treasury
        }
        
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "FUT"
        contract.exchange = exchange or exchange_map.get(symbol, "CME")
        contract.currency = currency
        contract.lastTradeDateOrContractMonth = UniversalContractBuilder._get_next_contract_month()
        
        return contract
    
    @staticmethod
    def _build_options_contract(symbol: str, strike: float, expiry: str, 
                               right: str = "C", exchange: str = "SMART", **kwargs) -> Contract:
        """Build options contract"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = exchange
        contract.currency = "USD"
        contract.strike = strike
        contract.right = right  # "C" for Call, "P" for Put
        contract.lastTradeDateOrContractMonth = expiry
        return contract
    
    @staticmethod
    def _build_crypto_contract(symbol: str, exchange: str = "PAXOS", **kwargs) -> Contract:
        """Build crypto contract"""
        if "." in symbol:
            base, quote = symbol.split(".")
        else:
            base = symbol.replace("USD", "")
            quote = "USD"
        
        contract = Contract()
        contract.symbol = base
        contract.secType = "CRYPTO"
        contract.exchange = exchange
        contract.currency = quote
        return contract
    
    @staticmethod
    def _get_next_contract_month() -> str:
        """Get next quarterly contract month"""
        now = datetime.now()
        quarters = ["03", "06", "09", "12"]
        
        current_quarter = (now.month - 1) // 3
        next_quarter = (current_quarter + 1) % 4
        
        year = now.year
        if next_quarter == 0 and current_quarter == 3:
            year += 1
        
        return f"{year}{quarters[next_quarter]}"
