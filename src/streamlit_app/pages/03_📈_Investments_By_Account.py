import argparse
import datetime
import logging
from pathlib import Path

import streamlit as st
from beancount import loader
from beancount.core import getters, prices

from beangrow import config as configlib
from beangrow import investments, reports


def main():

    if 'ledger' not in st.session_state:
        st.write('### Run Main Page First')
        return

    account_data_map = st.session_state.ledger.account_data_map
    dcontext = st.session_state.ledger.options_map['dcontext']

    account = st.sidebar.selectbox(
        'Select Account', [ad.account for ad in account_data_map.values()]
    )
    ad = account_data_map[account]
    investments.write_account_file(dcontext, ad)


if __name__ == '__main__':
    main()
