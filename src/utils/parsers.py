import re
import json
from typing import Dict, Any, Union
from .exceptions import ValidationError


def convert_str_to_dict(data: str) -> Dict[str, Any]:
    """
    Convert a string containing hexadecimal values to a dictionary.
    Converts hex values (0x...) to decimal before parsing JSON.

    Args:
        data: String containing JSON with possible hex values

    Returns:
        Dict containing parsed data with hex values converted to decimal

    Raises:
        ValidationError: If string cannot be parsed or is invalid JSON

    Example:
        >>> data = '{"cmd": 0x01, "value": 0x20}'
        >>> convert_str_to_dict(data)
        {'cmd': 1, 'value': 32}
    """
    try:
        # Convert '0x' hexadecimal values to decimal
        def convert_hex_to_dec(match):
            hex_value = match.group(0)
            return str(int(hex_value, 16))

        # Replace hex values with decimal
        input_string = re.sub(r'0[xX][0-9A-Fa-f]+', convert_hex_to_dec, data)

        # Parse the modified string as JSON
        return json.loads(input_string)

    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON format: {str(e)}")
    except ValueError as e:
        raise ValidationError(f"Invalid hexadecimal value: {str(e)}")
    except Exception as e:
        raise ValidationError(f"Error parsing string: {str(e)}")


def convert_dict_to_hex_str(data: Dict[str, Any], hex_keys: set = {'cmd'}) -> str:
    """
    Convert a dictionary back to a string with specified values as hex.

    Args:
        data: Dictionary to convert
        hex_keys: Set of keys whose values should be converted to hex

    Returns:
        JSON string with specified values in hex format

    Example:
        >>> data = {'cmd': 1, 'value': 32}
        >>> convert_dict_to_hex_str(data, {'cmd', 'value'})
        '{"cmd": 0x1, "value": 0x20}'
    """

    def convert_value(key: str, value: Any) -> Union[str, Any]:
        if key in hex_keys and isinstance(value, (int, str)):
            try:
                num = int(value)
                return f"0x{num:02x}"
            except (ValueError, TypeError):
                return value
        return value

    converted = {
        k: convert_value(k, v)
        for k, v in data.items()
    }

    return json.dumps(converted)


# Utility functions for command code conversion
def hex_to_int(hex_str: str) -> int:
    """Convert hex string to integer."""
    try:
        return int(hex_str, 16) if isinstance(hex_str, str) else hex_str
    except ValueError:
        raise ValidationError(f"Invalid hex value: {hex_str}")


def int_to_hex(value: int) -> str:
    """Convert integer to hex string."""
    return f"0x{value:02x}"


def convert_dict_to_str(data: Dict[str, Any]) -> str:
    """
    Convert a dictionary to JSON string.
    
    Args:
        data: Dictionary to convert
        
    Returns:
        JSON string representation
    """
    return json.dumps(data)


# Example usage and tests
if __name__ == "__main__":
    # Test conversion from string to dict
    test_str = '{"cmd": 0x01, "request_data": ["test.jpg"], "flags": 0x20}'
    result = convert_str_to_dict(test_str)
    print(f"Converted to dict: {result}")

    # Test conversion back to hex string
    test_dict = {'cmd': 1, 'request_data': ['test.jpg'], 'flags': 32}
    hex_str = convert_dict_to_hex_str(test_dict, {'cmd', 'flags'})
    print(f"Converted to hex string: {hex_str}")

    # Test hex/int conversion
    hex_value = "0x20"
    int_value = hex_to_int(hex_value)
    print(f"Hex to int: {hex_value} -> {int_value}")
    print(f"Int to hex: {int_value} -> {int_to_hex(int_value)}")