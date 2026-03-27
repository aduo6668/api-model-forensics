from .config import ensure_runtime_dirs
from .ui import run_app


def main() -> None:
    ensure_runtime_dirs()
    run_app()


if __name__ == "__main__":
    main()
