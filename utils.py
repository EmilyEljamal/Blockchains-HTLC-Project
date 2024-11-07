from typing import List, Dict

# Assuming utils and routing modules are available in Python
from utils import *
from routing import *

def is_equal_result(a: NodePairResult, b: NodePairResult) -> bool:
    return a.to_node_id == b.to_node_id

def is_equal_key_result(key: int, a: NodePairResult) -> bool:
    return key == a.to_node_id

def is_equal_long(a: int, b: int) -> bool:
    return a == b

def is_key_equal(a: Distance, b: Distance) -> bool:
    return a.node == b.node

def is_present(element: int, long_array: List[int]) -> bool:
    if long_array is None:
        return False
    
    return element in long_array

