import sys
from streamlit.web import cli as stcli
from pathlib import Path

def main():
    input_args = list(sys.argv[1:])
    returns = Path(__file__).parent.joinpath("01_🏠_Returns.py")
    sys.argv = ['streamlit', 'run', str(returns.resolve().absolute()), '--']
    sys.argv.extend(input_args)
    sys.exit(stcli.main())

if __name__ == '__main__':
    main()
