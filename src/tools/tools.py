import os
import platform
import json


def clear_terminal() -> None:
    """
    This function clear the terminal or command promt
    """
    plt = platform.platform()

    if plt.startswith("W"):  # platform is windows
        os.system("cls")
    else:  # platform is Linux or Mac
        os.system("clear")


def install_dependencies() -> None:
    os.system(f"pip install -r requirement.txt")


def read_from_file() -> dict:
    """
    Open json file which contains saved data.
    """

    with open('./data/data.json', 'r') as f:
        data = json.load(f)
    return data


def write_to_file(data) -> None:
    """
    Get data and write it into a json file.
    (use for writing inputs data)
    """
    with open("./data/data.json", "w") as f:
        json.dump(data, f)


class BiggerThanOneDescriptor:
    """
    A Descriptor class for preventing a class parameter get value lower than 1
    """

    def __init__(self):
        self.name = None

    def __set_name__(self, instance, name):
        self.name = name

    def __set__(self, instance: object, value: float):
        if value < 1:
            value = 1
        instance.__dict__[self.name] = value

    def __get__(self, instance: object, owner: type) -> float:
        if instance is None:
            return self
        return instance.__dict__[self.name]


class BetweenOneAndZero:
    """
    A Descriptor class for preventing a class parameter get value bigger than 1
    and lower than zero
    """

    def __init__(self):
        self.name = None

    def __set_name__(self, instance: object, name: str) -> None:
        self.name = name

    def __set__(self, instance: object, value: float):
        if value > 1:
            value = 1
        elif value < 0:
            value = 0
        instance.__dict__[self.name] = value

    def __get__(self, instance: object, owner: type) -> float:
        if instance is None:
            return self
        return instance.__dict__[self.name]
