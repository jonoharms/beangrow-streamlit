from beancount.core import prices
from beangrow import reports
import streamlit as st
from beangrow.returns import Pricer, Returns
from beangrow import returns as returnslib
from beangrow import investments
from beangrow import reports
from beancount.core import data
import pandas as pd


def main():

    st.write('# Cash Flow')

    if 'args' not in st.session_state:
        st.write('### Run Main Page First')
        return

    args = st.session_state.args

    # Generate output reports.
    entries = st.session_state.entries
    account_data_map = st.session_state.account_data_map
    config = st.session_state.config
    end_date = st.session_state.end_date

    """Write out returns report to a directory with files in it."""

    # TOOD(blais): Prices should be plot separately, by currency.
    # fprint("<h2>Prices</h2>")
    # pairs = set((r.currency, r.cost_currency) for r in account_data)
    # plots = plot_prices(dirname, pricer.price_map, pairs)
    # for _, filename in sorted(plots.items()):
    #     fprint('<img src={} style="width: 100%"/>'.format(filename))


if __name__ == '__main__':
    main()
