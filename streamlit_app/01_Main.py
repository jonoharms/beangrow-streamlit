#!/usr/bin/env python3
"""Calculate my true returns, including dividends and real costs.
"""

__copyright__ = 'Copyright (C) 2020  Martin Blais'
__license__ = 'GNU GPLv2'

import argparse
import datetime
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from beancount import loader
from beancount.core import data, getters, prices

from beangrow import config as configlib
from beangrow import investments, reports, streamlit_helpers
from beangrow import returns as returnslib
from beangrow.returns import Pricer


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

    with st.expander('More Info'):
        st.write(f'**Ledger:** {args.ledger.resolve()}')
        st.write(f'**Config:** {args.config.resolve()}')

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
    # fig = reports.plot_cumulative_flows(
    #     cash_flows, dates_all, gamounts, value_dates, value_values
    # )
    # st.write(fig)

    fig = px.line(st.session_state.values_df)
    # fig.update_xaxes(range=[df1.index[0], df1.index[-1]])
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig)

    st.write(st.session_state.returns.total)
    st.write(st.session_state.returns.exdiv)
    st.write(st.session_state.returns.div)
    st.write(st.session_state.calendar_returns)
    st.write(st.session_state.cumulative_returns)
    st.write(st.session_state.accounts_df)


if __name__ == '__main__':
    main()
