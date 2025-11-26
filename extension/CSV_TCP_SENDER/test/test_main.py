"""
main.py 함수들에 대한 단위 테스트
"""
import pytest
import os
import sys
import yaml
import json
import socket
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# main.py를 import하기 위해 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    ConfigError,
    CSVError,
    NetworkError,
    load_config_yaml,
    read_csv_file,
    is_valid_csv_file,
    make_ext_data_json,
    send_json_to_gateway,
    load_cache,
    save_cache,
    process_node_data,
    send_data_if_available,
    calculate_sleep_time,
)


class TestConfigLoading:
    """설정 파일 로드 테스트"""
    
    def test_load_valid_config(self, config_file, sample_config):
        """정상적인 설정 파일 로드"""
        gateway_config, node_configs, sender_config = load_config_yaml(config_file)
        
        assert gateway_config['ip'] == sample_config['gateway_config']['ip']
        assert gateway_config['port'] == sample_config['gateway_config']['port']
        assert len(node_configs) == 2
        assert sender_config['send_interval_seconds'] == 20
    
    def test_load_nonexistent_config(self):
        """존재하지 않는 설정 파일"""
        with pytest.raises(ConfigError) as exc_info:
            load_config_yaml('nonexistent.yaml')
        assert "설정 파일을 찾을 수 없습니다" in str(exc_info.value)
    
    def test_load_empty_config(self, temp_dir):
        """빈 설정 파일"""
        empty_config_path = os.path.join(temp_dir, 'empty.yaml')
        with open(empty_config_path, 'w', encoding='utf-8') as f:
            f.write('')
        
        with pytest.raises(ConfigError) as exc_info:
            load_config_yaml(empty_config_path)
        assert "설정 파일이 비어있습니다" in str(exc_info.value)
    
    def test_load_config_missing_gateway(self, temp_dir):
        """gateway_config가 없는 설정 파일"""
        config = {
            'node_config': [{'slave_id': '1'}],
            'sender_config': {'send_interval_seconds': 20}
        }
        config_path = os.path.join(temp_dir, 'no_gateway.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f)
        
        with pytest.raises(ConfigError) as exc_info:
            load_config_yaml(config_path)
        assert "gateway_config가 없습니다" in str(exc_info.value)
    
    def test_load_config_missing_node(self, temp_dir):
        """node_config가 없는 설정 파일"""
        config = {
            'gateway_config': {'ip': '127.0.0.1', 'port': 50020},
            'sender_config': {'send_interval_seconds': 20}
        }
        config_path = os.path.join(temp_dir, 'no_node.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f)
        
        with pytest.raises(ConfigError) as exc_info:
            load_config_yaml(config_path)
        assert "node_config가 없습니다" in str(exc_info.value)
    
    def test_load_config_missing_sender(self, temp_dir):
        """sender_config가 없는 설정 파일"""
        config = {
            'gateway_config': {'ip': '127.0.0.1', 'port': 50020},
            'node_config': [{'slave_id': '1'}]
        }
        config_path = os.path.join(temp_dir, 'no_sender.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f)
        
        with pytest.raises(ConfigError) as exc_info:
            load_config_yaml(config_path)
        assert "sender_config가 없습니다" in str(exc_info.value)


class TestCSVFileHandling:
    """CSV 파일 처리 테스트"""
    
    def test_read_valid_csv(self, csv_file):
        """정상적인 CSV 파일 읽기"""
        data = read_csv_file(csv_file)
        assert len(data) == 3
        assert data[0]['Temperature'] == '25.5'
        assert data[1]['Humidity'] == '59.5'
    
    def test_read_nonexistent_csv(self):
        """존재하지 않는 CSV 파일"""
        with pytest.raises(CSVError) as exc_info:
            read_csv_file('nonexistent.csv')
        assert "CSV 파일을 찾을 수 없습니다" in str(exc_info.value)
    
    def test_is_valid_csv_file(self, sample_csv_data):
        """CSV 파일 유효성 검사 - 정상"""
        assert is_valid_csv_file(sample_csv_data) == True
    
    def test_is_invalid_csv_file(self):
        """CSV 파일 유효성 검사 - 비정상"""
        invalid_data = [
            {'Datetime': '2025-10-01 10:'}  # 불완전한 날짜
        ]
        assert is_valid_csv_file(invalid_data) == False
    
    def test_read_invalid_csv_with_retry(self, invalid_csv_file):
        """유효하지 않은 CSV 파일 읽기 (재시도 후 실패)"""
        with pytest.raises(CSVError) as exc_info:
            read_csv_file(invalid_csv_file)
        assert "재시도 후에도 유효하지 않습니다" in str(exc_info.value)


class TestJSONGeneration:
    """JSON 생성 테스트"""
    
    def test_make_ext_data_json(self, sample_csv_data):
        """정상적인 JSON 생성"""
        ext_pos = '37.5665,126.9780'
        json_str = make_ext_data_json(sample_csv_data, ext_pos)
        
        # JSON 파싱 확인
        data = json.loads(json_str)
        assert data['id'] == 'extctrl'
        assert data['cmd'] == 'res_ext_data'
        assert len(data['data']) == 3
        assert data['data'][0]['ext_pos'] == ext_pos
        assert data['data'][0]['ext_temp'] == '25.5'
    
    def test_make_ext_data_json_with_timestamp(self, sample_csv_data):
        """timestamp를 지정한 JSON 생성"""
        ext_pos = '37.5665,126.9780'
        timestamp = '2025-10-01 12:00:00'
        json_str = make_ext_data_json(sample_csv_data, ext_pos, timestamp)
        
        data = json.loads(json_str)
        assert data['timestamp'] == timestamp
    
    def test_make_ext_data_json_empty_data(self):
        """빈 데이터로 JSON 생성"""
        json_str = make_ext_data_json([], '37.5665,126.9780')
        
        data = json.loads(json_str)
        assert len(data['data']) == 0


class TestNetworkCommunication:
    """네트워크 통신 테스트"""
    
    @patch('socket.socket')
    def test_send_json_to_gateway_success(self, mock_socket):
        """정상적인 데이터 전송"""
        mock_sock_instance = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock_instance
        
        json_str = '{"test": "data"}'
        send_json_to_gateway(json_str, '127.0.0.1', 50020)
        
        mock_sock_instance.connect.assert_called_once_with(('127.0.0.1', 50020))
        mock_sock_instance.sendall.assert_called_once()
    
    @patch('socket.socket')
    def test_send_json_to_gateway_timeout(self, mock_socket):
        """타임아웃 에러"""
        mock_sock_instance = MagicMock()
        mock_sock_instance.connect.side_effect = socket.timeout
        mock_socket.return_value.__enter__.return_value = mock_sock_instance
        
        with pytest.raises(NetworkError) as exc_info:
            send_json_to_gateway('{"test": "data"}', '127.0.0.1', 50020)
        assert "타임아웃" in str(exc_info.value)
    
    @patch('socket.socket')
    def test_send_json_to_gateway_connection_refused(self, mock_socket):
        """연결 거부 에러"""
        mock_sock_instance = MagicMock()
        mock_sock_instance.connect.side_effect = ConnectionRefusedError
        mock_socket.return_value.__enter__.return_value = mock_sock_instance
        
        with pytest.raises(NetworkError) as exc_info:
            send_json_to_gateway('{"test": "data"}', '127.0.0.1', 50020)
        assert "연결 거부됨" in str(exc_info.value)


class TestCacheManagement:
    """캐시 관리 테스트"""
    
    def test_load_existing_cache(self, cache_file, sample_cache):
        """기존 캐시 파일 로드"""
        with open(cache_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(sample_cache, f)
        
        cache = load_cache(cache_file)
        assert cache['test/data/slave_id_1.csv'] == '2025-10-01 10:02:00'
        assert cache['test/data/slave_id_2.csv'] == '2025-10-01 10:02:00'
    
    def test_load_nonexistent_cache(self, temp_dir):
        """존재하지 않는 캐시 파일 (빈 dict 반환)"""
        cache_path = os.path.join(temp_dir, 'nonexistent_cache.yaml')
        cache = load_cache(cache_path)
        assert cache == {}
    
    def test_save_cache(self, cache_file, sample_cache):
        """캐시 저장"""
        save_cache(cache_file, sample_cache)
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            loaded_cache = yaml.safe_load(f)
        
        assert loaded_cache == sample_cache


class TestNodeDataProcessing:
    """노드 데이터 처리 테스트"""
    
    def test_process_node_data_new_data(self, csv_file, empty_cache):
        """새로운 데이터 처리"""
        node_configs = [
            {
                'node_id': 'node_1',
                'csv_file_path': csv_file,
                'ext_pos': '37.5665,126.9780'
            }
        ]
        
        result = process_node_data(node_configs, empty_cache)
        
        assert len(result) == 3
        assert result[0]['ext_pos'] == '37.5665,126.9780'
        assert empty_cache['node_1'] == '2025-10-01 10:02:00'
    
    def test_process_node_data_with_cache(self, csv_file):
        """캐시를 사용한 데이터 처리 (필터링)"""
        cache = {'node_1': '2025-10-01 10:00:00'}
        node_configs = [
            {
                'node_id': 'node_1',
                'csv_file_path': csv_file,
                'ext_pos': '37.5665,126.9780'
            }
        ]
        
        result = process_node_data(node_configs, cache)
        
        # 10:00:00 이후 데이터만 반환 (10:01:00, 10:02:00)
        assert len(result) == 2
        assert cache['node_1'] == '2025-10-01 10:02:00'
    
    def test_process_node_data_no_csv_path(self, empty_cache):
        """csv_file_path가 없는 경우"""
        node_configs = [
            {
                'node_id': 'node_1',
                'ext_pos': '37.5665,126.9780'
                # csv_file_path 없음
            }
        ]
        
        result = process_node_data(node_configs, empty_cache)
        assert len(result) == 0


class TestDataSending:
    """데이터 전송 테스트"""
    
    @patch('main.send_json_to_gateway')
    @patch('main.save_cache')
    def test_send_data_if_available_with_data(self, mock_save_cache, mock_send, sample_csv_data, temp_dir):
        """데이터가 있을 때 전송"""
        # ext_pos 추가
        for row in sample_csv_data:
            row['ext_pos'] = '37.5665,126.9780'
        
        cache_file = os.path.join(temp_dir, 'cache.yaml')
        cache = {}
        
        send_data_if_available(sample_csv_data, '127.0.0.1', 50020, cache, cache_file)
        
        mock_send.assert_called_once()
        mock_save_cache.assert_called_once()
    
    @patch('main.send_json_to_gateway')
    @patch('main.save_cache')
    def test_send_data_if_available_no_data(self, mock_save_cache, mock_send, temp_dir):
        """데이터가 없을 때"""
        cache_file = os.path.join(temp_dir, 'cache.yaml')
        cache = {}
        
        send_data_if_available([], '127.0.0.1', 50020, cache, cache_file)
        
        mock_send.assert_not_called()
        mock_save_cache.assert_not_called()


class TestSleepTimeCalculation:
    """대기 시간 계산 테스트"""
    
    def test_calculate_sleep_time_normal(self):
        """정상적인 대기 시간 계산"""
        import time
        start_time = time.time()
        time.sleep(0.1)  # 0.1초 경과
        
        sleep_time = calculate_sleep_time(20, start_time)
        assert 19.8 <= sleep_time <= 20.0
    
    def test_calculate_sleep_time_negative(self):
        """음수 대기 시간 (0으로 반환)"""
        import time
        start_time = time.time() - 30  # 30초 전
        
        sleep_time = calculate_sleep_time(20, start_time)
        assert sleep_time == 0
    
    def test_calculate_sleep_time_none_interval(self):
        """send_interval_seconds가 None일 때"""
        import time
        start_time = time.time()
        
        sleep_time = calculate_sleep_time(None, start_time)
        assert sleep_time >= 19.9  # 기본값 20초 사용


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

