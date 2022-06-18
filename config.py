import yaml


class Config:

    def __init__(self, cfgFn):
        with open(cfgFn) as f:
            dataMap = yaml.safe_load(f)

        for key, value in dataMap.items():
            setattr(self, key, value)