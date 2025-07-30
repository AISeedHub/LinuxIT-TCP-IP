# Test

## Overview

This project includes:
- Model and protocol tests for a TCP server and detector.

## Running Tests

### 1. Model and Protocol Tests

Navigate to the `test/` directory and run:

```
pytest -xvs
```

This will execute:
- `test_model_detection.py` — Model loading, inference, and config tests.
- `test_overall_system.py` — System integration and runner tests.
- `test_protocol_requests.py` — TCP protocol and error handling tests.

You can run a specific test file:

```
pytest test/test_model_detection.py
```

### 2. Sensor App Test

From the project root, run:

```
python sensor_app/test_sensor_app.py
```

This will:
- Load the sensor configuration.
- Initialize the sensor model and handler.
- Send a sample request and verify the response.

## Configuration

- Edit `configs.py` for model and image paths.
- Edit `sensor_app/sensor_config.yaml` for sensor app settings.

## Notes

- Ensure all required model files and test images exist in the specified directories.
- The TCP server tests expect the server to be available on `127.0.0.1:8888`.

## Logging

- All tests and the sensor app use Python logging for output.
- Check the console for detailed info and error messages.

---

For further details, see the docstrings in each test file.
```