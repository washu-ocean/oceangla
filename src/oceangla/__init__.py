from .config import config
from .parser import parse_args
from pprint import pprint


def main():
    parse_args()
    pprint(config.__dict__)


if __name__ == "__main__":
    main()
