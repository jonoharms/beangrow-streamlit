#!/usr/bin/env python3
"""Calculate my true returns, including dividends and real costs.
"""

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn
import streamlit as st
from beancount.core import convert, data, prices
from beancount.core.amount import Amount
from beancount.parser import printer

from beangrow import investments
from beangrow import returns as returnslib
from beangrow.config_pb2 import (
    Config,  # type: ignore
    Group,  # type: ignore
)
from beangrow.investments import AccountData, CashFlow
from beangrow.returns import Pricer, Returns

__copyright__ = 'Copyright (C) 2020  Martin Blais'
__license__ = 'GNU GPLv2'

import collections
import datetime
import io
import logging
import multiprocessing
import os
import platform
import subprocess
import tempfile
import typing
from functools import partial
from os import path
from typing import Any, Optional, Tuple, Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas
from dateutil.relativedelta import relativedelta
from pandas.plotting import register_matplotlib_converters
from attrs import define

register_matplotlib_converters()

seaborn.set()


# Basic type aliases.
Account = str
Currency = str
Date = datetime.date
Array = np.ndarray


# The date at which we evaluate this.
TODAY = Date.today()

CurrencyPair = Tuple[Currency, Currency]

IRR_FORMAT = '{:32}: {:6.2%} ({:6.2%} ex-div, {:6.2%} div)'

# A named date interval: (name, start date, end date).
@define
class Interval:
    name: str
    start_date: Date
    end_date: Date


def compute_returns_table(
    pricer: Pricer,
    target_currency: Currency,
    account_data: list[AccountData],
    intervals: list[Interval],
):
    """Compute a table of sequential returns, and return the raw Returns objects as well."""

    returns = []
    for interval in intervals:
        cash_flows = returnslib.truncate_and_merge_cash_flows(
            pricer, account_data, interval.start_date, interval.end_date
        )
        ret = returnslib.compute_returns(
            cash_flows,
            pricer,
            target_currency,
            interval.end_date,
            groupname=interval.name,
        )
        returns.append(ret)

    return returns


def get_accounts_table(account_data: list[AccountData]) -> pandas.DataFrame:
    """Build of table of per-account information."""
    header = ['Investment', 'Description', 'Status']
    rows = []
    for ad in account_data:
        if ad.close is not None:
            status_str = 'Closed'
        elif ad.balance.is_empty():
            status_str = 'Empty'
        else:
            status_str = 'Active'

        name = (
            ad.commodity.meta.get('name', ad.commodity.currency)
            if ad.commodity
            else 'N/A'
        )
        rows.append((ad.account, name, status_str))
    return pandas.DataFrame(data=rows, columns=header)


def set_axis(ax_, date_min, date_max):
    """Setup X axis for dates."""

    years = mdates.YearLocator()
    years_fmt = mdates.DateFormatter('%Y')
    months = mdates.MonthLocator()

    ax_.xaxis.set_major_locator(years)
    ax_.xaxis.set_major_formatter(years_fmt)
    ax_.xaxis.set_minor_locator(months)

    if date_min and date_max:
        datemin = np.datetime64(date_min, 'Y')
        datemax = np.datetime64(date_max, 'Y') + np.timedelta64(1, 'Y')
        ax_.set_xlim(datemin, datemax)

    ax_.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax_.format_ydata = '{:,}'.format
    ax_.grid(True)


def plot_prices(
    output_dir: str, price_map: prices.PriceMap, pairs: list[CurrencyPair]
) -> dict[str, str]:
    """Render one or more plots of prices."""

    # Group by quote currencies.
    outplots = {}
    series = collections.defaultdict(list)
    for currency, quote_currency in pairs:
        series[quote_currency].append(currency)

    fig, axs = plt.subplots(
        len(series), 1, sharex=True, figsize=[10, 2 * len(series)]
    )
    if len(series) == 1:
        axs = [axs]
    for index, (qc, currencies) in enumerate(sorted(series.items())):
        ax = axs[index]
        for currency in currencies:
            price_points = prices.get_all_prices(price_map, (currency, qc))

            # Render cash flows.
            dates = [date for date, _ in price_points]
            prices_ = [float(price) for _, price in price_points]

            set_axis(
                ax, dates[0] if dates else None, dates[-1] if dates else None
            )
            ax.plot(dates, prices_, linewidth=0.3)
            ax.scatter(dates, prices_, s=1.2)

    fig.autofmt_xdate()
    fig.tight_layout()
    filename = outplots['price'] = path.join(output_dir, 'price.svg')
    plt.savefig(filename)
    plt.close(fig)

    return outplots


def get_amortized_value_plot_data_from_flows(
    price_map, flows, returns_rate, target_currency, dates
):
    date_min = dates[0] - datetime.timedelta(days=1)
    date_max = dates[-1]
    num_days = (date_max - date_min).days
    dates_all = [
        dates[0] + datetime.timedelta(days=x) for x in range(num_days)
    ]
    gamounts = np.zeros(num_days)
    rate = (1 + returns_rate) ** (1.0 / 365)

    for flow in flows:
        remaining_days = (date_max - flow.date).days
        if target_currency:
            conv_amount = convert.convert_amount(
                flow.amount, target_currency, price_map, date=flow.date
            ).number
            amt = -float(conv_amount) if conv_amount else 0.0
        else:
            amt = -float(flow.amount.number)
        if remaining_days > 0:
            gflow = amt * (rate ** np.arange(0, remaining_days))
            gamounts[-remaining_days:] += gflow
        else:
            if flow.source != 'simulated-close':
                gamounts[-1] += amt
    return dates_all, gamounts


def plot_flows_pyplot(flows: list[CashFlow]):

    dates = [f.date for f in flows]
    dates_exdiv = [f.date for f in flows if not f.is_dividend]
    dates_div = [f.date for f in flows if f.is_dividend]

    amounts_exdiv = np.array(
        [f.amount.number for f in flows if not f.is_dividend]
    )
    amounts_div = np.array([f.amount.number for f in flows if f.is_dividend])

    fig, axs = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=[10, 4],
        gridspec_kw={'height_ratios': [3, 1]},
    )
    for ax in axs:
        set_axis(ax, dates[0] if dates else None, dates[-1] if dates else None)
        ax.axhline(0, color='#000', linewidth=0.2)
        ax.vlines(
            np.array(dates_exdiv),
            0,
            amounts_exdiv,
            linewidth=3,
            color='#000',
            alpha=0.7,
        )
        ax.vlines(
            np.array(dates_div),
            0,
            amounts_div,
            linewidth=3,
            color='#0A0',
            alpha=0.7,
        )
    axs[1].set_yscale('symlog')

    axs[0].set_title('Cash Flows')
    axs[1].set_title('log(Cash Flows)')
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


def plot_flows_plotly(flows: pandas.DataFrame, log_plot: bool = True):
    df = flows.copy()
    df['log'] = [
        np.log10(amount) if amount > 0 else -np.log10(np.abs(amount))
        for amount in df['amount']
    ]
    data = 'log' if log_plot else 'amount'

    fig = px.bar(
        df,
        x='date',
        y=data,
        barmode='overlay',
        color='is_dividend',
        labels={
            'date': 'Date',
            'amount': 'Cash Flow Amount ($)',
            'log': 'Cash Flow Amount ($)',
            'is_dividend': 'Dividend',
        },
        hover_name='investment',
        hover_data={'date': True, 'amount': ':.2f', 'is_dividend': True},
    )
    fig.update_traces(width=1e9)

    if log_plot:
        vals = np.array(
            [1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 5e4, 1e5, 5e5, 1e5, 1e6]
        )
        vals = np.append(np.flip(-1.0 * vals), vals)
        vals = [val for val in vals if val < df['amount'].max()]
        vals = [val for val in vals if val > df['amount'].min()]
        log_vals = [
            np.log10(val) if val > 0 else -np.log10(np.abs(val))
            for val in vals
        ]

        text = [f'${val:.0f}' for val in vals]
        fig.update_yaxes(ticktext=text, tickvals=log_vals)

    fig.update_layout(height=600)
    return fig


def plot_cumulative_flows(
    flows, dates_all, gamounts, value_dates, value_values
):

    # Render cumulative cash flows, with returns growth.
    lw = 0.8
    dates = [f.date for f in flows]

    fig, ax = plt.subplots(figsize=[10, 4])
    ax.set_title('Cumulative value')
    set_axis(ax, dates[0] if dates else None, dates[-1] if dates else None)
    ax.axhline(0, color='#000', linewidth=lw)

    # ax.scatter(dates_all, gamounts, color='#000', alpha=0.2, s=1.0)
    ax.plot(dates_all, gamounts, color='#000', alpha=0.7, linewidth=lw)

    # Overlay value of assets over time.
    ax.plot(value_dates, value_values, color='#00F', alpha=0.5, linewidth=lw)
    ax.scatter(value_dates, value_values, color='#00F', alpha=lw, s=2)
    ax.legend(
        ['Amortized value from flows', 'Market value'], fontsize='xx-small'
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


def write_price_directives(
    filename: str, pricer: Pricer, days_price_threshold: int
):
    """Write a list of required price directives as a Beancount file."""
    price_entries = []
    for (currency, required_date), found_dates in sorted(
        pricer.required_prices.items()
    ):
        assert len(found_dates) == 1
        cost_currency, actual_date, rate = found_dates.pop()
        days_late = (required_date - actual_date).days
        if days_late < days_price_threshold:
            continue
        price = data.Price(
            {}, required_date, currency, Amount(rate, cost_currency)
        )
        price_entries.append(price)
    os.makedirs(path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as prfile:
        printer.print_entries(price_entries, file=prfile)


def get_calendar_intervals(date: Date) -> list[Interval]:
    """Return a list of date pairs for sequential intervals."""
    intervals = [
        Interval(str(year), Date(year, 1, 1), Date(year + 1, 1, 1))
        for year in range(TODAY.year - 15, TODAY.year)
    ]
    intervals.append(Interval(str(TODAY.year), Date(TODAY.year, 1, 1), date))
    return intervals


def get_cumulative_intervals(date: Date) -> list[Interval]:
    """Return a list of date pairs for sequential intervals."""
    return [
        Interval('15_years_ago', Date(date.year - 15, 1, 1), date),
        Interval('10_years_ago', Date(date.year - 10, 1, 1), date),
        Interval('5_years_ago', Date(date.year - 5, 1, 1), date),
        Interval('4_years_ago', Date(date.year - 4, 1, 1), date),
        Interval('3_years_ago', Date(date.year - 3, 1, 1), date),
        Interval('2_years_ago', Date(date.year - 2, 1, 1), date),
        Interval('1_year_ago', Date(date.year - 1, 1, 1), date),
        Interval('ytd', Date(date.year, 1, 1), date),
        Interval('rolling_6_months_ago', date - relativedelta(months=6), date),
        Interval('rolling_3_months_ago', date - relativedelta(months=3), date),
    ]


def generate_price_page(base_quote, price_map):

    all_prices = prices.get_all_prices(price_map, base_quote)
    if not all_prices:
        return None

    dates = np.array([date for date, _ in all_prices])
    prices_ = np.array([price for _, price in all_prices])

    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    set_axis(ax, dates[0], dates[-1])
    ax.set_title('Prices for {} ({})'.format(*base_quote))
    ax.plot(dates, prices_, linewidth=0.5)
    ax.scatter(dates, prices_, s=2.0)
    fig.autofmt_xdate()

    return fig
