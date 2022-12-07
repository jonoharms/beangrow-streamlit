import sys
from streamlit.web import cli as stcli

if __name__ == '__main__':

    input_args = list(sys.argv[1:])
    sys.argv = ['streamlit', 'run', '01_🏠_Returns.py', '--']
    sys.argv.extend(input_args)
    sys.exit(stcli.main())
