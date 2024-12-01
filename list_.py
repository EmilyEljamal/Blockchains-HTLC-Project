class Element:
    def __init__(self, data):
        self.data = data
        self.next = None

def push(head, data):
    new_head = Element(data)
    new_head.next = head
    return new_head

def get_by_key(head, key, is_key_equal):
    iterator = head
    while iterator is not None:
        if is_key_equal(key, iterator.data):
            return iterator.data
        iterator = iterator.next
    return None

def pop(head):
    if head is None:
        return None, None
    data = head.data
    new_head = head.next
    return new_head, data

def list_len(head):
    length = 0
    iterator = head
    while iterator is not None:
        length += 1
        iterator = iterator.next
    return length

def is_in_list(head, data, is_equal):
    iterator = head
    while iterator is not None:
        if is_equal(iterator.data, data):
            return True
        iterator = iterator.next
    return False

def list_free(lst):
    """Frees the list resources by clearing its contents."""
    lst.clear()
