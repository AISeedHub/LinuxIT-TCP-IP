import socket
import sys
from datetime import datetime
import os

BUFFER_SIZE = 65535 # max size of recv

DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 50020
LOG_DIR = os.path.join(os.path.dirname(__file__), './logs')
LOG_FILE = os.path.join(LOG_DIR, 'virtual_gateway.log')

def log_received_data(data):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'[{now}] {data}\n')

def start_tcp_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"[INFO] TCP/IP server is waiting at {host}:{port}")
        while True:
            client_socket, addr = server_socket.accept()
            with client_socket:
                print(f"[INFO] connected: {addr}")
                data = b''
                while True:
                    chunk = client_socket.recv(BUFFER_SIZE)
                    if not chunk:
                        break
                    data += chunk
                try:
                    decoded = data.decode('utf-8')
                    print(f"[RECV] {decoded}")
                    log_received_data(decoded)
                except Exception as e:
                    print(f"[ERROR] data decoding failed: {e}")
            print(f"[INFO] disconnected: {addr}")

if __name__ == "__main__":
    # with command argument
    # example: python virtual_gateway.py 0.0.0.0 50020
    if len(sys.argv) == 3:
        host = sys.argv[1]
        port = int(sys.argv[2])
    else:
        host = DEFAULT_HOST
        port = DEFAULT_PORT
    start_tcp_server(host, port)
