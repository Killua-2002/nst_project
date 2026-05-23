import config
from src.utils import print_data_status


def main():
    config.ensure_project_dirs()
    print("Created/checked project folders.")
    print_data_status()


if __name__ == "__main__":
    main()
