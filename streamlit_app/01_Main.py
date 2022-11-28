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
from streamlit_extras.dataframe_explorer import dataframe_explorer

from beangrow import streamlit_helpers

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
        'filter_reports',
        nargs='*',
        help='Optional names of specific subset of reports to analyze.',
    )

    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Verbose mode'
    )

    parser.add_argument(
        '-d',
        '--days-price-threshold',
        action='store',
        type=int,
        default=5,
        help='The number of days to tolerate price latency.',
    )

    parser.add_argument(
        '-e',
        '--end-date',
        action='store',
        type=datetime.date.fromisoformat,
        help='The end date to compute returns up to.',
    )

    parser.add_argument(
        '-j',
        '--parallel',
        action='store_true',
        help='Run report generation concurrently.',
    )

    parser.add_argument(
        '-E',
        '--check-explicit-flows',
        action='store_true',
        help=(
            'Enables comparison of the general categorization method '
            'with the explicit one with specialized explicit  handlers '
            'per signature.'
        ),
    )

    args = parser.parse_args()
    st.write('# Beangrow-Streamlit')
    st.write('Compute portfolio returns from a Beancount ledger')

    if 'args' not in st.session_state:

        if args.verbose:
            logging.basicConfig(
                level=logging.DEBUG, format='%(levelname)-8s: %(message)s'
            )
            logging.getLogger('matplotlib.font_manager').disabled = True

        streamlit_helpers.load_ledger(args)

    report = streamlit_helpers.select_report()
    if 'cash_flows' not in st.session_state:
        streamlit_helpers.load_report(report)

    fig = px.line(st.session_state.values_df)
    fig.update_layout(hovermode='x unified')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    columns = st.columns([1, 1, 3])
    rets = [
        st.session_state.returns.total,
        st.session_state.returns.exdiv,
        st.session_state.returns.div,
    ]
    for col, ret, name in zip(
        columns, rets, ['Total Return', 'Ex Dividends', 'With Dividends']
    ):
        col.metric(label=name, value=f'{ret*100:.3f}%')

    st.markdown('---')

    calendar_returns = st.session_state.calendar_returns[
        ['total', 'exdiv', 'div']
    ].multiply(100)
    calendar_returns = calendar_returns.loc[
        ~(calendar_returns == 0).all(axis=1)
    ]

    df = calendar_returns.transpose()
    st.write(df)

    cumulative_returns = st.session_state.cumulative_returns[
        ['total', 'exdiv', 'div']
    ].multiply(100)
    st.write(cumulative_returns.transpose())

    with st.expander('More Info'):
        st.write(f'**Ledger:** {args.ledger.resolve()}')
        st.write(f'**Config:** {args.config.resolve()}')
        tab0, tab1 = st.tabs(['Values DF', 'Accounts DF'])

        with tab0:
            st.write('### Value DF')
            tmp_df = dataframe_explorer(st.session_state.values_df)
            st.write(tmp_df)

        with tab1:
            st.write('### Accounts')
            tmp_df1 = dataframe_explorer(st.session_state.accounts_df)
            st.write(tmp_df1)


if __name__ == '__main__':
    main()
