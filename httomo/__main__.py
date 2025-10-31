import sys
from .cli import main

if __name__ == "__main__":
    standalone_mode = True
    if "--no-standalone" in sys.argv:
        standalone_mode = False
        sys.argv.remove("--no-standalone")

    main(standalone_mode=standalone_mode)
