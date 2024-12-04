def clear_():
    with open("output.txt", "w") as file:
        file.write("")

def print_(text):
    with open("output.txt", "a") as file:
        file.write(text)
        file.write("\n")