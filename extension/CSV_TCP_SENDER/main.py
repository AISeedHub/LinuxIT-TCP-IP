import yaml
import os
import csv
from datetime import datetime
import json
import time
import socket
import logging
from typing import Dict, List, Any, Optional, Tuple

# ========= Custom Exceptions =========
class ConfigError(Exception):
    """설정 파일 관련 에러"""
    pass

class CSVError(Exception):
    """CSV 파일 처리 관련 에러"""
    pass

class NetworkError(Exception):
    """네트워크 통신 관련 에러"""
    pass

# ========= Data Logger =========
def setup_logging() -> logging.Logger:
    """로깅 설정을 초기화합니다. 중복 설정을 방지합니다."""
    logger_name = "csv_tcp_sender"
    logger = logging.getLogger(logger_name)
    
    # 이미 핸들러가 설정되어 있으면 중복 설정 방지
    if logger.handlers:
        return logger
    
    # Set Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Set Basic Config for Logging (console output)
    logging.basicConfig(
        level=logging.DEBUG,
        format=formatter._fmt,
        datefmt=formatter.datefmt,
    )

    # Logger File Handler
    SCRIPT_DIR = os.path.dirname(__file__)
    os.makedirs(os.path.join(SCRIPT_DIR, 'logs'), exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(SCRIPT_DIR, 'logs', 'csv_tcp_sender.log'), 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    
    return logger

# Initialize logger
logger = setup_logging()

def load_config_yaml(config_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """설정 파일을 로드하고 유효성을 검사합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML 파일 파싱 에러: {e}")
    except Exception as e:
        raise ConfigError(f"설정 파일 로드 중 에러 발생: {e}")
    
    if not config:
        raise ConfigError("설정 파일이 비어있습니다.")
    
    # Check if gateway_config is valid
    gateway_config = config.get('gateway_config', {})
    if not gateway_config or len(gateway_config) == 0:
        raise ConfigError("gateway_config가 없습니다. 다음 형식을 사용하세요:\ngateway_config:\n  ip: '127.0.0.1'\n  port: 50020")

    # Check if node_config is valid
    node_configs = config.get('node_config', [])
    if not node_configs or len(node_configs) == 0:
        raise ConfigError("node_config가 없습니다. 다음 형식을 사용하세요:\nnode_config:\n  - slave_id: '1'\n    csv_file_path: 'path/to/csv'\n    ext_pos: '37.5665,126.9780'")

    # Check if sender_config is valid
    sender_config = config.get('sender_config', {})
    if not sender_config or len(sender_config) == 0:
        raise ConfigError("sender_config가 없습니다. 다음 형식을 사용하세요:\nsender_config:\n  send_interval_seconds: 20")

    return gateway_config, node_configs, sender_config

def read_csv_file(csv_file_path: str) -> List[Dict[str, str]]:
    while True:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            csv_lines = list(csv.DictReader(csvfile))

        if is_valid_csv_file(csv_lines):
            return csv_lines
        else:
            logger.warning(f"CSV 파일이 아직 완전히 기록되지 않음. 1초 후 재시도: {csv_file_path}")
            time.sleep(1)
    
def is_valid_csv_file(csv_lines: List[Dict[str, str]]) -> bool:
    # To prevent confliction of reading and writing csv file.
    # ex: 2025-06-29 10:00:00 : True
    # ex: 2025-06-29 10: : False
    try:
        dt_str = csv_lines[-1]['Datetime'][:-1]
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return True
    except ValueError:
        return False

def make_ext_data_json(data_rows: List[Dict[str, str]], ext_pos: str, timestamp: Optional[str] = None) -> str:
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

def send_json_to_gateway(json_str: str, ip: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(json_str.encode('utf-8'))
        logger.info(f"Sent JSON to {ip}:{port} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_cache(cache_path: str) -> Dict[str, Any]:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def save_cache(cache_path: str, cache_dict: Dict[str, Any]) -> None:
    with open(cache_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cache_dict, f, allow_unicode=True)

def process_node_data(node_configs: List[Dict[str, Any]], cache: Dict[str, Any]) -> List[Dict[str, str]]:
    """각 노드의 CSV 데이터를 처리하고 새로운 데이터를 반환"""
    new_csv_lines_in_node = []
    
    for node in node_configs:
        ext_pos = node.get('ext_pos', '')
        csv_path = node.get('csv_file_path')
        node_id = node.get('node_id', csv_path)  # node_id가 없으면 파일 경로로 대체
        last_dt = cache.get(node_id)
        
        if csv_path:
            logger.info(f"Reading CSV: {csv_path}")
            csv_lines = read_csv_file(csv_path)
            logger.info(f"{len(csv_lines)} rows loaded.")
            
            # last_dt 이후의 데이터만 추출
            filtered_rows = []
            for row in csv_lines:
                row_dt = row.get('Datetime')
                if last_dt is None or (row_dt and row_dt > last_dt):
                    row['ext_pos'] = ext_pos
                    filtered_rows.append(row)
            
            logger.info(f"New {len(filtered_rows)} rows Founded")
            if filtered_rows:
                # 최신 Datetime으로 cache 업데이트
                cache[node_id] = filtered_rows[-1]['Datetime']
            new_csv_lines_in_node.extend(filtered_rows)
        else:
            logger.warning("csv_file_path not found in node config.")
    
    return new_csv_lines_in_node

def send_data_if_available(new_csv_lines_in_node: List[Dict[str, str]], gateway_ip: str, gateway_port: int, cache: Dict[str, Any], cache_file: str) -> None:
    """새로운 데이터가 있으면 전송하고 캐시 저장"""
    if new_csv_lines_in_node:
        logger.info(f"전송할 새로운 데이터가 {len(new_csv_lines_in_node)}개 있습니다.")
        # 마지막 노드의 ext_pos 사용 (모든 노드가 같은 위치라고 가정)
        ext_pos = new_csv_lines_in_node[-1].get('ext_pos', '') if new_csv_lines_in_node else ''
        json_str = make_ext_data_json(new_csv_lines_in_node, ext_pos)
        logger.info(f"전송시도: {gateway_ip}:{gateway_port}")
        send_json_to_gateway(json_str, gateway_ip, gateway_port)
        logger.info(f"전송완료: {gateway_ip}:{gateway_port}")
        save_cache(cache_file, cache)
    else:
        logger.info("전송할 새로운 데이터가 없습니다.")

def process_single_loop() -> Optional[int]:
    """단일 루프 실행"""
    startTime = time.time()
    logger.info("========== Start Loop ===========")
    
    try:
        # Load config.yaml
        CONFIG_FILE = os.path.join(SCRIPT_DIR, 'config.yaml')
        gateway_config, node_configs, sender_config = load_config_yaml(CONFIG_FILE)
        gateway_ip = gateway_config.get('ip')
        gateway_port = int(gateway_config.get('port'))
        send_interval_seconds = sender_config.get('send_interval_seconds')
        logger.info("config.yaml 로딩 완료")

        # Load .cache.yaml
        CACHE_FILE = os.path.join(SCRIPT_DIR, '.cache.yaml')
        cache = load_cache(CACHE_FILE)
        logger.info(".cache.yaml 로딩 완료")
        
        # Process node data
        new_csv_lines_in_node = process_node_data(node_configs, cache)
        
        # Send data if available
        send_data_if_available(new_csv_lines_in_node, gateway_ip, gateway_port, cache, CACHE_FILE)
        
        return send_interval_seconds
        
    except Exception as e:
        import traceback
        error_info = traceback.extract_tb(e.__traceback__)[-1]
        file_name = error_info.filename
        line_no = error_info.lineno
        logger.error(f"Error in {file_name} at line {line_no}: {str(e)}")
        return None

def calculate_sleep_time(send_interval_seconds: Optional[int], start_time: float) -> float:
    """다음 루프까지의 대기 시간 계산"""
    if not send_interval_seconds:
        send_interval_seconds = 20
        logger.warning(f"Send_interval_seconds not found in config, using default value: {send_interval_seconds} seconds")
    
    end_time = time.time()
    sleep_time = send_interval_seconds - (end_time - start_time)
    if sleep_time < 0:
        logger.warning("Sleep time is negative, using 0 seconds")
        sleep_time = 0
    
    return sleep_time

def main() -> None:
    """메인 실행 함수"""
    while True:
        start_time = time.time()
        send_interval_seconds = process_single_loop()
        
        # Calculate sleep time
        sleep_time = calculate_sleep_time(send_interval_seconds, start_time)
        
        # Sleep for next loop
        logger.info(f"Sleep For Next Loop: {sleep_time:.2f} seconds")
        time.sleep(sleep_time)






if __name__ == "__main__":
    main()
