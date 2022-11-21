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

import glob
import streamlit as st

st.set_page_config(layout="wide")

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

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format='%(levelname)-8s: %(message)s'
        )
        logging.getLogger('matplotlib.font_manager').disabled = True

    # Figure out end date.
    end_date = args.end_date or datetime.date.today()

    # Load the example file.
    logging.info('Reading ledger: %s', args.ledger)
    entries, _, options_map = loader.load_file(args.ledger)
    accounts = getters.get_accounts(entries)
    dcontext = options_map['dcontext']

    # Load, filter and expand the configuration.
    config = configlib.read_config(args.config, args.filter_reports, accounts)
    args.output.mkdir(exist_ok=True)

    with open(args.output.joinpath('config.pbtxt'), 'w') as efile:
        print(config, file=efile)

    cash_tab, investments_tab, by_type = st.tabs(['Cash Flows', 'Investments', 'Investments By Type'])

    account_data_map = investments.extract(
        entries,
        config,
        end_date,
        False
    )

    with investments_tab:
        # Write out a details file for each account for debugging.
        account = st.selectbox('Select Account', [ad.account for ad in account_data_map.values()])
        ad = account_data_map[account]
        investments.write_account_file(dcontext, ad)

    with by_type:
        # Output transactions for each type (for debugging).
        investments.write_transactions_by_type(account_data_map.values(), dcontext)

    # Generate output reports.
    output_reports = args.output.joinpath("groups")

    with cash_tab:
        st.text(f'Number of entries loaded: {len(entries)}')
        pricer = reports.generate_reports(account_data_map, config,
                                        prices.build_price_map(entries),
                                        end_date,
                                        output_reports,
                                        args.parallel, args.pdf)



if __name__ == '__main__':
    main()
