class Array:
    def __init__(self, size):
        self.size = size
        self.index = 0
        self.elements = [None] * size

    def _resize(self):
        self.size *= 2
        self.elements.extend([None] * (self.size - len(self.elements)))

    def insert(self, data):
        if self.index >= self.size:
            self._resize()
        self.elements[self.index] = data
        self.index += 1

    def get(self, i):
        if i >= self.size or i >= self.index:
            return None
        return self.elements[i]

    def length(self):
        return self.index

    def reverse(self):
        self.elements[:self.index] = self.elements[:self.index][::-1]

    def delete(self, element, is_equal):
        for i in range(self.index):
            if is_equal(self.elements[i], element):
                self._delete_element_at(i)
                break

    def _delete_element_at(self, element_index):
        for i in range(element_index, self.index - 1):
            self.elements[i] = self.elements[i + 1]
        self.index -= 1
        self.elements[self.index] = None

    def delete_all(self):
        self.index = 0

    def free(self):
        del self.elements

