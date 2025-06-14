
class d:
    def __init__(self):
        self.v=1

class lalala:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3
        self.d_obj = d()

from load_yaml import load_yaml_into_obj


test_obj = lalala()

test_obj = load_yaml_into_obj("sample.yaml", test_obj)
