class Heap:
    def __init__(self, size):
        self.size = size
        self.index = 0
        self.data = []

    def _resize(self):
        self.size *= 2

    def insert(self, data, compare):
        # Check if resizing is needed
        if self.index >= self.size:
            self._resize()

        # Add data to the heap and maintain the heap property manually
        self.data.append(data)
        self.index += 1
        self._sift_up(self.index - 1, compare)

    def insert_or_update(self, data, compare, is_key_equal):
        # Search for existing data with matching key
        for i in range(self.index):
            if is_key_equal(self.data[i], data):
                self.data[i] = data  # Update data
                self._sift_up(i, compare)  # Re-heapify up
                self._sift_down(i, compare)  # Re-heapify down
                return
        # If not found, insert as a new item
        self.insert(data, compare)

    def pop(self, compare):
        if self.index == 0:
            return None

        # Get the min element (root) and move the last element to the root
        min_element = self.data[0]
        self.index -= 1
        if self.index > 0:
            self.data[0] = self.data.pop()
            self._sift_down(0, compare)
        else:
            self.data.pop()

        return min_element

    def length(self):
        return self.index

    def free(self):
        self.data = []
        self.index = 0

    # Helper function to maintain heap property from child to parent
    def _sift_up(self, i, compare):
        parent = (i - 1) // 2
        while i > 0 > compare(self.data[i], self.data[parent]):
            self.data[i], self.data[parent] = self.data[parent], self.data[i]
            i = parent
            parent = (i - 1) // 2

    # Helper function to maintain heap property from parent to child
    def _sift_down(self, i, compare):
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i

            if left < self.index and compare(self.data[left], self.data[smallest]) < 0:
                smallest = left
            if right < self.index and compare(self.data[right], self.data[smallest]) < 0:
                smallest = right

            if smallest == i:
                break

            self.data[i], self.data[smallest] = self.data[smallest], self.data[i]
            i = smallest
