"""
통합 테스트 - 전체 프로세스 플로우 테스트
"""
import pytest
import os
import sys
import yaml
import csv
import socket
import threading
import time
from unittest.mock import patch, MagicMock

# main.py를 import하기 위해 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    process_single_loop,
    load_config_yaml,
    process_node_data,
    send_data_if_available,
    calculate_sleep_time,
)


class TestEndToEndFlow:
    """End-to-End 통합 테스트"""
    
    def test_full_process_with_new_data(self, temp_dir, sample_csv_data, sample_config):
        """새로운 데이터가 있을 때 전체 프로세스"""
        # 설정 파일 생성
        config_path = os.path.join(temp_dir, 'config.yaml')
        
        # CSV 파일 생성
        csv_path = os.path.join(temp_dir, 'test_data.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = sample_csv_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_csv_data)
        
        # 설정 업데이트
        sample_config['node_config'][0]['csv_file_path'] = csv_path
        sample_config['node_config'].pop()  # 두 번째 노드 제거
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(sample_config, f, allow_unicode=True)
        
        # 캐시 파일 경로
        cache_path = os.path.join(temp_dir, '.cache.yaml')
        
        # SCRIPT_DIR mock
        with patch('main.SCRIPT_DIR', temp_dir):
            with patch('main.send_json_to_gateway') as mock_send:
                interval = process_single_loop()
                
                assert interval == 20
                mock_send.assert_called_once()
                
                # 캐시가 저장되었는지 확인
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache = yaml.safe_load(f)
                    assert 'node_1' in cache or sample_config['node_config'][0]['csv_file_path'] in cache
    
    def test_full_process_with_cached_data(self, temp_dir, sample_csv_data, sample_config):
        """이미 처리된 데이터가 있을 때"""
        # 설정 파일 생성
        config_path = os.path.join(temp_dir, 'config.yaml')
        
        # CSV 파일 생성
        csv_path = os.path.join(temp_dir, 'test_data.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = sample_csv_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_csv_data)
        
        # 설정 업데이트
        sample_config['node_config'][0]['csv_file_path'] = csv_path
        sample_config['node_config'][0]['node_id'] = 'test_node'
        sample_config['node_config'].pop()  # 두 번째 노드 제거
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(sample_config, f, allow_unicode=True)
        
        # 캐시 파일 생성 (모든 데이터가 이미 처리됨)
        cache_path = os.path.join(temp_dir, '.cache.yaml')
        cache = {'test_node': '2025-10-01 10:02:00'}
        with open(cache_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cache, f)
        
        # SCRIPT_DIR mock
        with patch('main.SCRIPT_DIR', temp_dir):
            with patch('main.send_json_to_gateway') as mock_send:
                interval = process_single_loop()
                
                assert interval == 20
                # 새로운 데이터가 없으므로 전송되지 않음
                mock_send.assert_not_called()
    
    def test_full_process_with_config_error(self, temp_dir):
        """설정 파일 에러 처리"""
        # 잘못된 설정 파일 생성
        config_path = os.path.join(temp_dir, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: content: [[[')
        
        with patch('main.SCRIPT_DIR', temp_dir):
            interval = process_single_loop()
            assert interval is None  # 에러 발생시 None 반환
    
    def test_full_process_with_csv_error(self, temp_dir, sample_config):
        """CSV 파일 에러 처리"""
        # 설정 파일 생성
        config_path = os.path.join(temp_dir, 'config.yaml')
        
        # 존재하지 않는 CSV 파일 경로 설정
        sample_config['node_config'][0]['csv_file_path'] = 'nonexistent.csv'
        sample_config['node_config'].pop()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(sample_config, f, allow_unicode=True)
        
        with patch('main.SCRIPT_DIR', temp_dir):
            interval = process_single_loop()
            assert interval is None  # 에러 발생시 None 반환


class TestMultipleNodes:
    """다중 노드 처리 테스트"""
    
    def test_process_multiple_nodes(self, temp_dir):
        """여러 노드의 데이터 동시 처리"""
        # 노드 1 CSV
        csv_data_1 = [
            {'Temperature': '25.0', 'Humidity': '60.0', 'Rain': '0.0', 'Solar': '800.0', 'Wind_Direction': '180.0', 'Wind_Speed': '5.0', 'Datetime': '2025-10-01 10:00:00'},
            {'Temperature': '25.1', 'Humidity': '59.5', 'Rain': '0.0', 'Solar': '810.0', 'Wind_Direction': '185.0', 'Wind_Speed': '5.1', 'Datetime': '2025-10-01 10:01:00'}
        ]
        csv_path_1 = os.path.join(temp_dir, 'node1.csv')
        with open(csv_path_1, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data_1[0].keys())
            writer.writeheader()
            writer.writerows(csv_data_1)
        
        # 노드 2 CSV
        csv_data_2 = [
            {'Temperature': '26.0', 'Humidity': '55.0', 'Rain': '0.0', 'Solar': '850.0', 'Wind_Direction': '190.0', 'Wind_Speed': '6.0', 'Datetime': '2025-10-01 10:00:00'},
            {'Temperature': '26.1', 'Humidity': '54.5', 'Rain': '0.0', 'Solar': '860.0', 'Wind_Direction': '195.0', 'Wind_Speed': '6.1', 'Datetime': '2025-10-01 10:01:00'}
        ]
        csv_path_2 = os.path.join(temp_dir, 'node2.csv')
        with open(csv_path_2, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data_2[0].keys())
            writer.writeheader()
            writer.writerows(csv_data_2)
        
        # 노드 설정
        node_configs = [
            {'csv_file_path': csv_path_1, 'ext_pos': '37.5665,126.9780'},
            {'csv_file_path': csv_path_2, 'ext_pos': '37.5670,126.9785'}
        ]
        
        cache = {}
        result = process_node_data(node_configs, cache)
        
        # 총 4개의 데이터 (각 노드에서 2개씩)
        assert len(result) == 4
        assert cache[csv_path_1] == '2025-10-01 10:01:00'
        assert cache[csv_path_2] == '2025-10-01 10:01:00'


class TestConcurrency:
    """동시성 관련 테스트"""
    
    def test_sleep_time_calculation_accuracy(self):
        """대기 시간 계산의 정확성"""
        import time
        
        start_time = time.time()
        time.sleep(0.5)  # 0.5초 대기
        
        sleep_time = calculate_sleep_time(10, start_time)
        
        # 약 9.5초 남음
        assert 9.4 <= sleep_time <= 9.6
    
    def test_sleep_time_with_long_processing(self):
        """처리 시간이 인터벌보다 긴 경우"""
        import time
        
        start_time = time.time() - 15  # 15초 전에 시작했다고 가정
        
        sleep_time = calculate_sleep_time(10, start_time)
        
        # 음수가 아닌 0 반환
        assert sleep_time == 0


class TestErrorRecovery:
    """에러 복구 테스트"""
    
    @patch('main.send_json_to_gateway')
    def test_network_error_recovery(self, mock_send, temp_dir, sample_csv_data, sample_config):
        """네트워크 에러 후 복구"""
        from main import NetworkError
        
        # 설정 파일 생성
        config_path = os.path.join(temp_dir, 'config.yaml')
        
        # CSV 파일 생성
        csv_path = os.path.join(temp_dir, 'test_data.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = sample_csv_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_csv_data)
        
        # 설정 업데이트
        sample_config['node_config'][0]['csv_file_path'] = csv_path
        sample_config['node_config'].pop()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(sample_config, f, allow_unicode=True)
        
        # 네트워크 에러 발생하도록 mock 설정
        mock_send.side_effect = NetworkError("연결 실패")
        
        with patch('main.SCRIPT_DIR', temp_dir):
            interval = process_single_loop()
            
            # 에러가 발생해도 프로세스는 계속 진행 (None 반환)
            assert interval is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

