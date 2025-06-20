import argparse
import ruamel.yaml as ryaml
import os
import sys
import pathlib

dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))
import tools

def load_config(config_path, config_list=None):
    yaml = ryaml.YAML(typ="safe", pure=True)
    configs = yaml.load(pathlib.Path(config_path).read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *config_list] if config_list else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    return parser.parse_args([])
if __name__ == "__main__":
    config = load_config("/home/kensuke/PytorchReachability/configs.yaml")
    print(config.dt)
