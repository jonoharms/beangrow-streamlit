#!/usr/bin/env python3
"""Calculate my true returns, including dividends and real costs.
"""

__copyright__ = 'Copyright (C) 2020  Martin Blais'
__license__ = 'GNU GPLv2'


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

from beangrow.returns import Pricer
from beangrow import returns as returnslib
from beancount.core import data

import streamlit as st
import pandas as pd

import plotly.express as px

st.set_page_config(layout="wide")

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
        'output',
        help='Output directory to write all output files to.',
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
        '--pdf',
        '--pdfs',
        action='store_true',
        help='Render as PDFs. Default is HTML directories.',
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

    st.write("# Beangrow")
    args = parser.parse_args()
    end_date = args.end_date or datetime.date.today()
    if 'args' not in st.session_state:
        st.session_state.args = args

        if args.verbose:
            logging.basicConfig(
                level=logging.DEBUG, format='%(levelname)-8s: %(message)s'
            )
            logging.getLogger('matplotlib.font_manager').disabled = True

        # Figure out end date.
        

        # Load the example file.
        logging.info('Reading ledger: %s', args.ledger)
        entries, _, options_map = loader.load_file(args.ledger)
        accounts = getters.get_accounts(entries)
        dcontext = options_map['dcontext']

        # Load, filter and expand the configuration.
        config = configlib.read_config(args.config, args.filter_reports, accounts)

        st.session_state.entries = entries
        st.session_state.accounts = accounts
        st.session_state.options_map = options_map
        st.session_state.config = config
        st.session_state.end_date = end_date

        st.session_state.account_data_map = investments.extract(
            entries,
            config,
            end_date,
            False
        )
        st.success("Finished Reading Ledger")
        st.text(f'Number of entries loaded: {len(entries)}')


    report = st.sidebar.selectbox('Group', st.session_state.config.groups.group, format_func=lambda x: x.name)
    price_map = prices.build_price_map(st.session_state.entries)
    pricer = Pricer(price_map)
    account_data = [st.session_state.account_data_map[name] for name in report.investment]

    target_currency = report.currency
    if not target_currency:
        cost_currencies = set(r.cost_currency for r in account_data)
        target_currency = cost_currencies.pop()
        assert not cost_currencies, (
            "Incompatible cost currencies {} for accounts {}".format(
                cost_currencies, ",".join([r.account for r in account_data])))

    cash_flows = returnslib.truncate_and_merge_cash_flows(pricer, account_data,
                                                            None, end_date)
    
    returns = returnslib.compute_returns(cash_flows, pricer, target_currency, end_date)

    transactions = data.sorted([txn for ad in account_data for txn in ad.transactions])

    # Render cash flows.
    show_pyplot = st.sidebar.checkbox('Show pyplot plot', False)
    if show_pyplot:
        fig = reports.plot_flows_pyplot(cash_flows)
        st.write(fig)

    log_plot = st.sidebar.checkbox('Log Plot', True)
    df = investments.cash_flows_to_table(cash_flows)
    fig = reports.plot_flows_plotly(df, log_plot)
    st.plotly_chart(fig)
    st.write(df)

    dates = [f.date for f in cash_flows]
    dates_all, gamounts = reports.get_amortized_value_plot_data_from_flows(price_map, cash_flows, returns.total, target_currency, dates)
    value_dates, value_values = returnslib.compute_portfolio_values(price_map, transactions, target_currency)
    df1 = pd.DataFrame(index=dates_all, data=gamounts, columns= ['cumvalue'])


    fig = reports.plot_cumulative_flows(cash_flows, dates_all, gamounts, value_dates, value_values)
    df2 = pd.DataFrame(index=value_dates, data=value_values, columns= ['prices'])   
    df = pd.concat([df1, df2], axis=1).sort_index().astype(float)

    st.write(fig)
    fig = px.line(df)
    fig.update_xaxes(range=[df1.index[0],df1.index[-1]])
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig)

    st.write(returns.total)
    st.write(returns.exdiv)
    st.write(returns.div)

    table = reports.compute_returns_table(pricer, target_currency, account_data,
                                   reports.get_calendar_intervals(TODAY))
    st.write(table)

    table = reports.compute_returns_table(pricer, target_currency, account_data,
                                  reports.get_cumulative_intervals(TODAY))
    st.write(table)

    accounts_df = reports.get_accounts_table(account_data)
    st.write(accounts_df)

    

if __name__ == '__main__':
    main()
