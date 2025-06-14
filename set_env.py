import os
import sys

def set_mt5_env():
    # MT5 credentials
    os.environ['MT5_LOGIN'] = "13387082"
    os.environ['MT5_PASSWORD'] = "3d67ab!O"
    os.environ['MT5_SERVER'] = "ErranteSC-Demo"
    os.environ['MT5_PATH'] = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    
    print("MT5 environment variables set successfully!")
    print(f"Login: {os.environ.get('MT5_LOGIN')}")
    print(f"Server: {os.environ.get('MT5_SERVER')}")

if __name__ == "__main__":
    set_mt5_env() 