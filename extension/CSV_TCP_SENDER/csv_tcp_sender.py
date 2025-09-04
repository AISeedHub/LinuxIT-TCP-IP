import yaml
import os
import csv
from datetime import datetime
import json
import time
import socket

def load_config_yaml(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    gateway_config = config.get('gateway_config', {})
    node_configs = config.get('node_config', [])
    sender_config = config.get('sender_config', {})
    return gateway_config, node_configs, sender_config

def read_csv_file(csv_file_path):
    while True:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            csv_lines = list(csv.DictReader(csvfile))

        if is_valid_csv_file(csv_lines):
            return csv_lines
        else:
            print(f"[WARN] CSV 파일이 아직 완전히 기록되지 않음. 1초 후 재시도: {csv_file_path}")
            time.sleep(1)
    
def is_valid_csv_file(csv_lines):
    # To prevent confliction of reading and writing csv file.
    # ex: 2025-06-29 10:00:00 : True
    # ex: 2025-06-29 10: : False
    try:
        dt_str = csv_lines[-1]['Datetime'][:-1]
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return True
    except ValueError:
        return False

def make_ext_data_json(data_rows, ext_pos, timestamp=None):
    """
    data_rows: list of dict, 각 dict는 센서 데이터 한 줄
    ext_pos: str, 위치 정보 (예: '37.5665,126.9780')
    timestamp: str, 최종 timestamp (없으면 현재 시각)
    """
    data_list = []
    for row in data_rows:
        data = {
            "ext_pos": ext_pos,
            "ext_temp": str(row.get("Temperature", -1.0)),
            "ext_hum": str(row.get("Humidity", -1.0)),
            "ext_rainfall": str(row.get("Rain", -1.0)),
            "ext_solar": str(row.get("Solar", -1.0)),
            # "ext_irr_amount": str(row.get("Irr_Amount", -1.0)),
            "ext_w_dir": str(row.get("Wind_Direction", -1.0)),
            "ext_w_spd": str(row.get("Wind_Speed", -1.0)),
            "r_time": row.get("Datetime", str(-1.0)),
        }
        data_list.append(data)

    jsonDict = dict()
    jsonDict["id"] = "extctrl" # fixed
    jsonDict["cmd"] = "res_ext_data" # fixed
    jsonDict["data"] = data_list
    jsonDict["timestamp"] = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S") # current time (sending time)
    
    return json.dumps(jsonDict, ensure_ascii=False, indent=4)

def send_json_to_gateway(json_str, ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(json_str.encode('utf-8'))
        print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Sent JSON to {ip}:{port} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def save_cache(cache_path, cache_dict):
    with open(cache_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cache_dict, f, allow_unicode=True)

if __name__ == "__main__":
    while True:
        startTime = time.time()
        print()
        CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.yaml')
        print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - config.yaml 로딩 완료")
        CACHE_FILE = os.path.join(os.path.dirname(__file__), '.cache.yaml')
        print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - .cache.yaml 로딩 완료")
        gateway_config, node_configs, sender_config = load_config_yaml(CONFIG_FILE)
        gateway_ip = gateway_config.get('ip')
        gateway_port = int(gateway_config.get('port'))
        cache = load_cache(CACHE_FILE)
        send_interval_seconds = sender_config.get('send_interval_seconds', 10)
        new_csv_lines_in_node = []
        for node in node_configs:
            ext_pos = node.get('ext_pos', '')
            csv_path = node.get('csv_file_path')
            node_id = node.get('node_id', csv_path)  # node_id가 없으면 파일 경로로 대체
            last_dt = cache.get(node_id)
            if csv_path:
                print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Reading CSV: {csv_path}")
                csv_lines = read_csv_file(csv_path)
                print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {len(csv_lines)} rows loaded.")
                # last_dt 이후의 데이터만 추출
                filtered_rows = []
                for row in csv_lines:
                    row_dt = row.get('Datetime')
                    if last_dt is None or (row_dt and row_dt > last_dt):
                        row['ext_pos'] = ext_pos
                        filtered_rows.append(row)
                if filtered_rows:
                    # 최신 Datetime으로 cache 업데이트
                    cache[node_id] = filtered_rows[-1]['Datetime']
                new_csv_lines_in_node.extend(filtered_rows)
            else:
                print("[WARN] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - csv_file_path not found in node config.")
        # JSON 생성 및 전송
        if new_csv_lines_in_node:
            print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 전송할 새로운 데이터가 {len(new_csv_lines_in_node)}개 있습니다.")
            json_str = make_ext_data_json(new_csv_lines_in_node, ext_pos)
            send_json_to_gateway(json_str, gateway_ip, gateway_port)
            save_cache(CACHE_FILE, cache)
        else:
            print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 전송할 새로운 데이터가 없습니다.")
        
        endTime = time.time()
        sleepTime = send_interval_seconds - (endTime - startTime)
        print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - sleepTime: {sleepTime:.2f} seconds")
        time.sleep(sleepTime)

