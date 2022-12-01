#!/usr/bin/env python3
"""Calculate my true returns, including dividends and real costs.
"""

__copyright__ = 'Copyright (C) 2020  Martin Blais'
__license__ = 'GNU GPLv2'

import argparse
import datetime
import logging
from pathlib import Path

import plotly.express as px
import streamlit as st
from beangrow import returns as returnslib
from streamlit_extras.dataframe_explorer import dataframe_explorer

from beangrow.streamlit_helpers import Ledger

st.set_page_config(layout='wide')

Date = datetime.date
TODAY = Date.today()


def main():
    """Top-level function."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'ledger',
        help='Beancount ledger file',
        type=Path,
    )

    parser.add_argument(
        'config',
        action='store',
        help='Configuration for accounts and reports.',
        type=Path,
    )

    parser.add_argument(
        '-j',
        '--parallel',
        action='store_true',
        help='Run report generation concurrently.',
    )

    args = parser.parse_args()
    st.write('# Beangrow-Streamlit')
    st.write('Compute portfolio returns from a Beancount ledger')

    if 'ledger' not in st.session_state:
        ledger = Ledger.from_args(args)
        st.session_state['ledger'] = ledger
    else:
        ledger = st.session_state.ledger

    report = ledger.select_report()
    if 'reportdata' not in st.session_state:
        reportdata = ledger.load_report(report)
        st.session_state['reportdata'] = reportdata
    else:
        reportdata = st.session_state.reportdata

    fig = reportdata.plot_plotly()
    st.plotly_chart(fig)

    columns = st.columns([1, 1, 3])
    rets = [
        reportdata.returns.total,
        reportdata.returns.exdiv,
        reportdata.returns.div,
    ]
    for col, ret, name in zip(
        columns, rets, ['Total Return', 'Ex Dividends', 'With Dividends']
    ):
        col.metric(label=name, value=f'{ret*100:.3f}%')

    st.markdown('---')

    calendar_returns = returnslib.returns_to_dataframe(
        reportdata.calendar_returns
    )
    calendar_returns = calendar_returns[['total', 'exdiv', 'div']].multiply(
        100
    )
    calendar_returns = calendar_returns.loc[
        ~(calendar_returns == 0).all(axis=1)
    ]

    df = calendar_returns.transpose()
    st.write(df)

    cumulative_returns = returnslib.returns_to_dataframe(
        reportdata.cumulative_returns
    )
    cumulative_returns = cumulative_returns[
        ['total', 'exdiv', 'div']
    ].multiply(100)
    st.write(cumulative_returns.transpose())

    with st.expander('More Info'):
        st.write(f'**Ledger:** {args.ledger.resolve()}')
        st.write(f'**Config:** {args.config.resolve()}')
        tab0, tab1 = st.tabs(['Values DF', 'Accounts DF'])

        with tab0:
            st.write('### Value DF')
            tmp_df = dataframe_explorer(reportdata.portfolio_value)
            st.write(tmp_df)

        with tab1:
            st.write('### Accounts')
            tmp_df1 = dataframe_explorer(reportdata.accounts)
            st.write(tmp_df1)


if __name__ == '__main__':
    main()
