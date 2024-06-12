from abc import ABC, abstractmethod
import yaml
from functools import lru_cache

class UtilYAML(ABC):

    @property
    @abstractmethod
    def classpath(self):
        pass

    # @lru_cache(maxsize=None)
    @staticmethod
    def load_yaml_file(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error while parsing YAML file: {file_path}\n{e}")
