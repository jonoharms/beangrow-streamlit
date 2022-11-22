import argparse
import datetime
import logging
from pathlib import Path

from beancount import loader
from beancount.core import getters
from beancount.core import prices

from beangrow import investments
from beangrow import reports
from beangrow import config as configlib

import streamlit as st

def main():

    if 'args' not in st.session_state:
        st.write('### Run Main Page First')
        return
        
    args = st.session_state.args
    account_data_map = st.session_state.account_data_map
    dcontext = st.session_state.options_map['dcontext']

    account = st.sidebar.selectbox('Select Account', [ad.account for ad in account_data_map.values()])
    ad = account_data_map[account]
    investments.write_account_file(dcontext, ad)

if __name__ == '__main__':
    main()