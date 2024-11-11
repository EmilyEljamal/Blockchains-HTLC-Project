from enum import Enum
from heap_ import Heap  # Importing the Heap class

class EventType(Enum):
    FINDPATH = 1
    SENDPAYMENT = 2
    FORWARDPAYMENT = 3
    RECEIVEPAYMENT = 4
    FORWARDSUCCESS = 5
    FORWARDFAIL = 6
    RECEIVESUCCESS = 7
    RECEIVEFAIL = 8
    OPENCHANNEL = 9

class Event:
    def __init__(self, time, event_type, node_id, payment):
        self.time = time
        self.type = event_type
        self.node_id = node_id
        self.payment = payment

    def __lt__(self, other):
        return self.time < other.time

def new_event(time, event_type, node_id, payment):
    return Event(time, event_type, node_id, payment)

def compare_event(e1, e2):
    if e1.time == e2.time:
        return 0
    return -1 if e1.time < e2.time else 1

def initialize_events(payments_array):
    events_heap = Heap(size=payments_array.length() * 10)  # Initializing the heap based on Array size
    for i in range(payments_array.length()):
        payment = payments_array.get(i)
        event = new_event(payment.start_time, EventType.FINDPATH, payment.sender, payment)
        events_heap.insert(event, compare_event)
    return events_heap
