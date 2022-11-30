import datetime
import logging

import pandas as pd
import streamlit as st
from beancount import loader
from beancount.core import data, getters, prices

from beangrow import config as configlib
from beangrow import investments, reports
from beangrow import returns as returnslib
from beangrow.returns import Pricer
from attrs import define
from typing import NamedTuple
from beangrow.config_pb2 import (
    Config,
)

Date = datetime.date
TODAY = Date.today()


@define
class Ledger:

    entries: list[NamedTuple]
    options_map: dict
    accounts: list[str]
    end_date: datetime.date
    config: Config
    pricer: Pricer
    price_map: prices.PriceMap
    account_data_map: dict

    @classmethod
    def from_args(cls, args):
        # Load the example file.
        # logging.info('Reading ledger: %s', args.ledger)
        entries, _, options_map = loader.load_file(args.ledger)
        accounts = list(getters.get_accounts(entries))

        end_date = args.end_date or datetime.date.today()

        # Load, filter and expand the configuration.
        config = configlib.read_config(
            args.config, args.filter_reports, list(accounts)
        )

        account_data_map = investments.extract(
            entries, config, end_date, False
        )

        price_map = prices.build_price_map(entries)
        pricer = Pricer(price_map)

        return cls(
            entries,
            options_map,
            accounts,
            end_date,
            config,
            pricer,
            price_map,
            account_data_map,
        )

    def select_report(self):
        if 'report' not in st.session_state:
            current_report_index = 0
        else:
            current_report = st.session_state.report
            tmp = [
                config.name == current_report.name
                for config in self.config.groups.group
            ]
            current_report_index = next(i for i, rep in enumerate(tmp) if rep)

        report = st.sidebar.selectbox(
            'Group',
            self.config.groups.group,
            format_func=lambda x: x.name,
            index=current_report_index,
        )

        reload_button = st.sidebar.button(
            'Load Group',
            on_click=self.load_report,
            args=(report,),
        )

        return report

    def load_report(self, report):

        account_data = [
            self.account_data_map[name] for name in report.investment
        ]

        target_currency = report.currency

        if not target_currency:
            cost_currencies = set(r.cost_currency for r in account_data)
            target_currency = cost_currencies.pop()
            assert (
                not cost_currencies
            ), 'Incompatible cost currencies {} for accounts {}'.format(
                cost_currencies, ','.join([r.account for r in account_data])
            )

        cash_flows = returnslib.truncate_and_merge_cash_flows(
            self.pricer, account_data, None, self.end_date
        )

        returns = returnslib.compute_returns(
            cash_flows,
            self.pricer,
            target_currency,
            self.end_date,
        )

        transactions = data.sorted(
            [txn for ad in account_data for txn in ad.transactions]
        )

        dates = [f.date for f in cash_flows]
        dates_all, gamounts = reports.get_amortized_value_plot_data_from_flows(
            self.price_map,
            cash_flows,
            returns.total,
            target_currency,
            dates,
        )
        value_dates, value_values = returnslib.compute_portfolio_values(
            self.price_map, transactions, target_currency
        )

        df1 = pd.DataFrame(
            index=dates_all, data=gamounts, columns=['cumvalue']
        )
        df2 = pd.DataFrame(
            index=value_dates, data=value_values, columns=['prices']
        )
        values_df = pd.concat([df1, df2], axis=1).sort_index().astype(float)
        values_df = values_df[values_df['cumvalue'].notna()]

        calendar_returns = reports.compute_returns_table(
            self.pricer,
            target_currency,
            account_data,
            reports.get_calendar_intervals(TODAY),
        )

        cumulative_returns = reports.compute_returns_table(
            self.pricer,
            target_currency,
            account_data,
            reports.get_cumulative_intervals(TODAY),
        )

        accounts_df = reports.get_accounts_table(account_data)

        st.session_state.cash_flows = cash_flows
        st.session_state.returns = returns
        st.session_state.transactions = transactions
        st.session_state.values_df = values_df
        st.session_state.calendar_returns = calendar_returns
        st.session_state.cumulative_returns = cumulative_returns
        st.session_state.accounts_df = accounts_df
        st.session_state.report = report
