import pathlib

SAVE_PATH = pathlib.Path(__file__).parent.parent / "saves"
SAVE_PATH.mkdir(parents=True, exist_ok=True)

PROJECT_ADAPTER_SAVE_FILE = SAVE_PATH / "project_dump.pkl"

STATE_MANAGER_SAVE_FILE_DIR = SAVE_PATH / "states"
STATE_MANAGER_SAVE_FILE_DIR.mkdir(parents=True, exist_ok=True)

STATE_MANAGER_SAVE_FILE_TEMPLATE = "saved_state_{num_saves}.pkl"
