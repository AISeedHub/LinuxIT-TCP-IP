# CSV TCP Sender 테스트

이 디렉토리는 `main.py`의 모든 기능에 대한 테스트를 포함합니다.

## 테스트 구조

```
test/
├── __init__.py              # 패키지 초기화
├── conftest.py              # pytest fixtures 정의
├── test_main.py             # 단위 테스트
├── test_integration.py      # 통합 테스트
└── README.md                # 이 파일
```


## 테스트 실행 방법

### 1. 모든 테스트 실행
```bash
uv run pytest -v
```


## 테스트 클래스 설명

### test_main.py - 단위 테스트

1. **TestConfigLoading**: 설정 파일 로딩 테스트
   - 정상 설정 파일 로드
   - 존재하지 않는 파일 처리
   - 잘못된 설정 파일 처리

2. **TestCSVFileHandling**: CSV 파일 처리 테스트
   - CSV 파일 읽기
   - 유효성 검사
   - 재시도 로직

3. **TestJSONGeneration**: JSON 생성 테스트
   - 데이터 -> JSON 변환
   - 타임스탬프 처리

4. **TestNetworkCommunication**: 네트워크 통신 테스트
   - 정상 전송
   - 타임아웃 처리
   - 연결 거부 처리

5. **TestCacheManagement**: 캐시 관리 테스트
   - 캐시 로드/저장
   - 빈 캐시 처리

6. **TestNodeDataProcessing**: 노드 데이터 처리 테스트
   - 새로운 데이터 처리
   - 캐시 필터링
   - 다중 노드 처리

7. **TestDataSending**: 데이터 전송 테스트
   - 조건부 전송 로직

8. **TestSleepTimeCalculation**: 대기 시간 계산 테스트
   - 정상 계산
   - 음수 처리
   - 기본값 처리

### test_integration.py - 통합 테스트

1. **TestEndToEndFlow**: End-to-End 테스트
   - 전체 프로세스 플로우
   - 에러 처리 플로우

2. **TestMultipleNodes**: 다중 노드 테스트
   - 여러 노드 동시 처리

3. **TestConcurrency**: 동시성 테스트
   - 타이밍 관련 테스트

4. **TestErrorRecovery**: 에러 복구 테스트
   - 네트워크 에러 복구
