from typing import Optional
from array_ import Array

def is_equal_result(a, b) -> bool:
    """Check if two node pair results are equal."""
    return a.to_node_id == b.to_node_id

def is_equal_key_result(key, a) -> bool:
    """Check if a key matches the `to_node_id` in a node pair result."""
    return key == a.to_node_id

def is_equal_long(a, b) -> bool:
    """Check if two integers are equal."""
    return a == b

def is_key_equal(a, b) -> bool:
    """Check if two Distance objects have the same node."""
    return a.node == b.node

def is_present(element: int, long_array: Optional[Array]) -> bool:
    """Check if an element is present in an Array of integers."""
    if long_array is None:
        return False

    for i in range(long_array.length()):
        if long_array.get(i) == element:
            return True
    return False
