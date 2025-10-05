"""
pytest 설정 및 공통 fixtures
"""
import pytest
import os
import yaml
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """임시 디렉토리를 생성하고 테스트 후 삭제"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """테스트용 설정 데이터"""
    return {
        'gateway_config': {
            'ip': '127.0.0.1',
            'port': 50020
        },
        'node_config': [
            {
                'slave_id': '1',
                'csv_file_path': 'test/data/slave_id_1.csv',
                'ext_pos': '37.5665,126.9780'
            },
            {
                'slave_id': '2',
                'csv_file_path': 'test/data/slave_id_2.csv',
                'ext_pos': '37.5665,126.9780'
            }
        ],
        'sender_config': {
            'send_interval_seconds': 20
        }
    }


@pytest.fixture
def config_file(temp_dir, sample_config):
    """임시 설정 파일 생성"""
    config_path = os.path.join(temp_dir, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(sample_config, f, allow_unicode=True)
    return config_path


@pytest.fixture
def cache_file(temp_dir):
    """임시 캐시 파일 경로"""
    return os.path.join(temp_dir, '.cache.yaml')


@pytest.fixture
def sample_csv_data():
    """테스트용 CSV 데이터"""
    return [
        {
            'Temperature': '25.5',
            'Humidity': '60.0',
            'Rain': '0.0',
            'Solar': '800.0',
            'Wind_Direction': '180.0',
            'Wind_Speed': '5.2',
            'Datetime': '2025-10-01 10:00:00'
        },
        {
            'Temperature': '25.6',
            'Humidity': '59.5',
            'Rain': '0.0',
            'Solar': '810.0',
            'Wind_Direction': '185.0',
            'Wind_Speed': '5.5',
            'Datetime': '2025-10-01 10:01:00'
        },
        {
            'Temperature': '25.7',
            'Humidity': '59.0',
            'Rain': '0.0',
            'Solar': '820.0',
            'Wind_Direction': '190.0',
            'Wind_Speed': '5.8',
            'Datetime': '2025-10-01 10:02:00'
        }
    ]


@pytest.fixture
def csv_file(temp_dir, sample_csv_data):
    """임시 CSV 파일 생성"""
    import csv
    
    csv_path = os.path.join(temp_dir, 'test_data.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if sample_csv_data:
            fieldnames = sample_csv_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_csv_data)
    
    return csv_path


@pytest.fixture
def invalid_csv_file(temp_dir):
    """유효하지 않은 CSV 파일 생성 (마지막 행이 불완전)"""
    import csv
    
    csv_path = os.path.join(temp_dir, 'invalid_data.csv')
    
    data = [
        {
            'Datetime': '2025-10-01 10:00:00',
            'Temperature': '25.5',
            'Humidity': '60.0',
        },
        {
            'Datetime': '2025-10-01 10:',  # 불완전한 데이터
            'Temperature': '25.6',
            'Humidity': '59.5',
        }
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Datetime', 'Temperature', 'Humidity']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    return csv_path


@pytest.fixture
def sample_cache():
    """테스트용 캐시 데이터"""
    return {
        'test/data/slave_id_1.csv': '2025-10-01 10:02:00',
        'test/data/slave_id_2.csv': '2025-10-01 10:02:00'
    }


@pytest.fixture
def empty_cache():
    """빈 캐시 데이터"""
    return {}

