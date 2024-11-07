from typing import List, Dict
from utils import *
import routing as rt # type: ignore
import htlc as htlc # type: ignore

def is_equal_result(a: htlc.node_pair_result, b: htlc.node_pair_result) -> bool:
    return a.to_node_id == b.to_node_id

def is_equal_key_result(key: int, a: htlc.node_pair_result) -> bool:
    return key == a.to_node_id

def is_equal_long(a: int, b: int) -> bool:
    return a == b

def is_key_equal(a: rt.distance, b: rt.distance) -> bool:
    return a.node == b.node

def is_present(element: int, long_array: List[int]) -> bool:
    if long_array is None:
        return False
    
    return element in long_array

