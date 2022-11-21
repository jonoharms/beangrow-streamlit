#!/usr/bin/env python3
"""Calculate my true returns, including dividends and real costs.
"""

__copyright__ = "Copyright (C) 2020  Martin Blais"
__license__ = "GNU GPLv2"

from os import path
from typing import Any, Dict, List, Tuple, Optional
from functools import partial
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

from dateutil.relativedelta import relativedelta
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import seaborn
seaborn.set()

from beancount.core import data
from beancount.core import prices
from beancount.core.amount import Amount
from beancount.parser import printer
from beancount.core import convert

from beangrow.config_pb2 import Config, Group
from beangrow.investments import AccountData
from beangrow.investments import CashFlow
from beangrow import investments
from beangrow import returns as returnslib
from beangrow.returns import Pricer, Returns

import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

# Basic type aliases.
Account = str
Currency = str
Date = datetime.date
Array = np.ndarray


# The date at which we evaluate this.
TODAY = Date.today()


CurrencyPair = Tuple[Currency, Currency]


IRR_FORMAT = "{:32}: {:6.2%} ({:6.2%} ex-div, {:6.2%} div)"


STYLE = """
@media print {
    @page { margin: 0in; }
    body { margin: 0.2in; }
    .new-page { page-break-before: always; }
}

body, table { font: 9px Noto Sans, sans-serif; }

p { margin: 0.2em; }

table { border-collapse: collapse; }
table td, table th { border: thin solid black; }

table.full { width: 100%; }

/* p { margin-bottom: .1em } */
"""

RETURNS_TEMPLATE_PRE = """
<html>
  <head>
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans&display=swap" rel="stylesheet">

    <style>
      {style}
    </style>
  <head>
    <body>
      <h1>{title}</h1>
"""
RETURNS_TEMPLATE_POST = """
    </body>
</html>
"""


Table = typing.NamedTuple("Table", [("header", List[str]),
                                    ("rows", List[List[Any]])])

def render_table(table: Table,
                 floatfmt: Optional[str] = None,
                 classes: Optional[str] = None) -> str:
    """Render a simple data table to HTML."""
    oss = io.StringIO()
    fprint = partial(print, file=oss)
    fprint('<table class="{}">'.format(" ".join(classes or [])))
    fprint('<tr>')
    for heading in table.header:
        fprint("<th>{}</th>".format(heading))
    fprint('</tr>')
    for row in table.rows:
        fprint('<tr>')
        for value in row:
            if isinstance(value, float) and floatfmt:
                value = floatfmt.format(value)
            fprint("<td>{}</td>".format(value))
        fprint('</tr>')
    fprint("</table>")
    return oss.getvalue()


# A named date interval: (name, start date, end date).
Interval = Tuple[str, Date, Date]


def _compute_returns_with_table(pricer: Pricer,
                          target_currency: Currency,
                          account_data: List[AccountData],
                          intervals: List[Interval]):
    """Compute a table of sequential returns, and return the raw Returns objects as well."""
    header = ["Return"]
    rows = [["Total"], ["Ex-div"], ["Div"]]
    returns_list = []
    for intname, date1, date2 in intervals:
        header.append(intname)
        cash_flows = returnslib.truncate_and_merge_cash_flows(pricer, account_data,
                                                              date1, date2)
        returns = returnslib.compute_returns(cash_flows, pricer, target_currency, date2, groupname=intname)
        returns_list.append(returns)
        rows[0].append(returns.total)
        rows[1].append(returns.exdiv)
        rows[2].append(returns.div)
    return Table(header, rows), returns_list

def compute_returns_table(pricer: Pricer,
                          target_currency: Currency,
                          account_data: List[AccountData],
                          intervals: List[Interval]):
    """Compute a table of sequential returns."""
    return _compute_returns_with_table(pricer, target_currency, account_data, intervals)[0]


def write_returns_st(dirname: str,
                       pricer: Pricer,
                       account_data: List[AccountData],
                       title: str,
                       end_date: Date,
                       target_currency: Optional[Currency] = None):
    """Write out returns report to a directory with files in it."""

    logging.info("Writing returns dir for %s: %s", title, dirname)

    with open(path.join(dirname, "index.html"), "w") as indexfile:

        if not target_currency:
            cost_currencies = set(r.cost_currency for r in account_data)
            target_currency = cost_currencies.pop()
            assert not cost_currencies, (
                "Incompatible cost currencies {} for accounts {}".format(
                    cost_currencies, ",".join([r.account for r in account_data])))

        # TOOD(blais): Prices should be plot separately, by currency.
        # fprint("<h2>Prices</h2>")
        # pairs = set((r.currency, r.cost_currency) for r in account_data)
        # plots = plot_prices(dirname, pricer.price_map, pairs)
        # for _, filename in sorted(plots.items()):
        #     fprint('<img src={} style="width: 100%"/>'.format(filename))

        st.write("### Cash Flows")

        cash_flows = returnslib.truncate_and_merge_cash_flows(pricer, account_data,
                                                               None, end_date)
        
        returns = returnslib.compute_returns(cash_flows, pricer, target_currency, end_date)

        transactions = data.sorted([txn for ad in account_data for txn in ad.transactions])

        # # Note: This is where the vast majority of the time is spent.
        plots = plot_flows(dirname, pricer.price_map,
                            cash_flows, transactions, returns.total, target_currency)
        # fprint('<img src={} style="width: 100%"/>'.format(plots["flows"]))
        # fprint('<img src={} style="width: 100%"/>'.format(plots["cumvalue"]))

        # fprint("<h2>Returns</h2>")
        # fprint(render_table(Table(["Total", "Ex-Div", "Div"],
        #                           [[returns.total, returns.exdiv, returns.div]]),
        #                     floatfmt="{:.2%}"))

        # # Compute table of returns over intervals.
        # table = compute_returns_table(pricer, target_currency, account_data,
        #                               get_calendar_intervals(TODAY))
        # fprint("<p>", render_table(table, floatfmt="{:.1%}", classes=["full"]), "</p>")

        # table = compute_returns_table(pricer, target_currency, account_data,
        #                               get_cumulative_intervals(TODAY))
        # fprint("<p>", render_table(table, floatfmt="{:.1%}", classes=["full"]), "</p>")

        # fprint('<h2 class="new-page">Accounts</h2>')
        # fprint("<p>Report Currency: {}</p>".format(target_currency))
        # accounts_df = get_accounts_table(account_data)
        # fprint(accounts_df.to_html())

        # fprint('<h2 class="new-page">Cash Flows</h2>')
        # df = investments.cash_flows_to_table(cash_flows)
        # fprint(df.to_html())

        # fprint(RETURNS_TEMPLATE_POST)

    return




ReportData = typing.NamedTuple("ReportData", [
    ("cash_flows", pandas.DataFrame),
    ("returns", pandas.DataFrame),
    ("flow_value", pandas.Series),
    ("flow_amortized_value", pandas.Series),
    ("portfolio_value", pandas.Series),
    ("benchmark_func", callable),
])

def compute_report_data(pricer,
                       account_data,
                       end_date,
                       target_currency,
                       additional_cash_flows: Optional[List[Tuple[Date, Amount, Account]]]=None):
    
    additional_cash_flows = [CashFlow(d,amt,False,"additional",ac) for (d, amt, ac) in (additional_cash_flows or [])]
    cash_flows = flows = returnslib.truncate_and_merge_cash_flows(pricer, account_data, None, end_date, additional_cash_flows=additional_cash_flows)
    total_returns = returnslib.compute_returns(cash_flows, pricer, target_currency, end_date, groupname="Total")
    transactions = data.sorted([txn for ad in account_data for txn in ad.transactions])
    
    calendar_returns = _compute_returns_with_table(pricer, target_currency, account_data, get_calendar_intervals(end_date))[1]
    cumulative_returns = _compute_returns_with_table(pricer, target_currency, account_data, get_cumulative_intervals(end_date))[1]
    
    dates = [f.date for f in flows]

    dates_flow, amounts_flow = run_benchmark(cash_flows, dates, target_currency, pricer.price_map, 0)
    flow_value = pandas.Series(amounts_flow, index=dates_flow)

    dates_amortized, amounts_amortized = run_benchmark(cash_flows, dates, target_currency, pricer.price_map, total_returns.total) 
    flow_amortized_value = pandas.Series(amounts_amortized, index=dates_amortized)

    def benchmark_func(benchmark_commodity, additional_returns_rate):
        dates_benchmark, amounts_benchmark = run_benchmark(cash_flows, dates, target_currency, pricer.price_map, benchmark_commodity=benchmark_commodity, returns_rate=additional_returns_rate) 
        return pandas.Series(amounts_benchmark, index=dates_benchmark)
    
    dates_value, amounts_value = returnslib.compute_portfolio_values(pricer.price_map, transactions, target_currency)
    portfolio_value = pandas.Series(amounts_value, index=dates_value)

    header = ["date", "amount", "original_amount", "original_currency", "is_dividend", "source", "investment"]
    rows = []
    for flow in flows:
        if flow.source == 'simulated-close':
            continue
        amt = float(convert.convert_amount(flow.amount, target_currency, pricer.price_map, date=flow.date).number)
        rows.append((flow.date,
                    amt,
                    float(flow.amount.number),
                    flow.amount.currency,
                    flow.is_dividend,
                    flow.source,
                    flow.account))
    cash_flows_df = pandas.DataFrame(columns=header, data=rows)

    return ReportData(
        cash_flows_df,
        returnslib.returns_to_dataframe(calendar_returns+cumulative_returns+[total_returns]),
        flow_value,
        flow_amortized_value,
        portfolio_value,
        benchmark_func
    )

def write_returns_debugfile(filename: str,
                            pricer: Pricer,
                            account_data: List[AccountData],
                            title: str,
                            end_date: Date,
                            target_currency: Optional[Currency] = None) -> subprocess.Popen:
    """Write out returns report to a directory with files in it."""

    logging.info("Writing returns dir for %s: %s", title, filename)
    os.makedirs(path.dirname(filename), exist_ok=True)
    with open(filename, "w") as indexfile:
        fprint = partial(print, file=indexfile)
        fprint("* {}".format(title))
        fprint("** Cash Flows")
        fprint()
        cash_flows = returnslib.truncate_and_merge_cash_flows(pricer, account_data,
                                                              None, end_date)
        df = investments.cash_flows_to_table(cash_flows)
        fprint(df.to_string())

    return indexfile.name


def get_accounts_table(account_data: List[AccountData]) -> pandas.DataFrame:
    """Build of table of per-account information."""
    header = ["Investment", "Description", "Status"]
    rows = []
    for ad in account_data:
        if ad.close is not None:
            status_str = "Closed"
        elif ad.balance.is_empty():
            status_str = "Empty"
        else:
            status_str = "Active"

        name = (ad.commodity.meta.get("name", ad.commodity.currency)
                if ad.commodity else "N/A")
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
    ax_.format_ydata = "{:,}".format
    ax_.grid(True)


def plot_prices(output_dir: str,
                price_map: prices.PriceMap,
                pairs: List[CurrencyPair]) -> Dict[str, str]:
    """Render one or more plots of prices."""

    # Group by quote currencies.
    outplots = {}
    series = collections.defaultdict(list)
    for currency, quote_currency in pairs:
        series[quote_currency].append(currency)

    fig, axs = plt.subplots(len(series), 1, sharex=True, figsize=[10, 2 * len(series)])
    if len(series) == 1: axs = [axs]
    for index, (qc, currencies) in enumerate(sorted(series.items())):
        ax = axs[index]
        for currency in currencies:
            price_points = prices.get_all_prices(price_map, (currency, qc))

            # Render cash flows.
            dates = [date for date, _ in price_points]
            prices_ = [float(price) for _, price in price_points]

            set_axis(ax, dates[0] if dates else None, dates[-1] if dates else None)
            ax.plot(dates, prices_, linewidth=0.3)
            ax.scatter(dates, prices_, s=1.2)

    fig.autofmt_xdate()
    fig.tight_layout()
    filename = outplots["price"] = path.join(output_dir, "price.svg")
    plt.savefig(filename)
    plt.close(fig)

    return outplots

def run_benchmark(flows, dates, target_currency, price_map, returns_rate=None, benchmark_commodity=None):
    date_min = dates[0] - datetime.timedelta(days=1)
    date_max = dates[-1]
    num_days = (date_max - date_min).days
    dates_all = [dates[0] + datetime.timedelta(days=x) for x in range(num_days)]

    target_daily_return = (1 + returns_rate) ** (1./365) if returns_rate else 1

    if benchmark_commodity:
        bench_dates, bench_prices = zip(*prices.get_all_prices(price_map, (benchmark_commodity, target_currency)))
        bench_dates = pandas.DatetimeIndex(pandas.to_datetime(bench_dates))
        bench_prices = pandas.to_numeric(bench_prices)
        benchmark_df = pandas.DataFrame(
            data=bench_prices, 
            index=bench_dates, 
            columns=['price'])
        if benchmark_df.index.max() < pandas.to_datetime(date_max):
            benchmark_df.loc[pandas.to_datetime(date_max)] = benchmark_df.loc[benchmark_df.index.max()]
        benchmark_df = benchmark_df.resample("D")
        benchmark_df = benchmark_df.interpolate(method='linear')
        benchmark_df['daily_returns'] = benchmark_df['price'].pct_change()
        
    benchmark_current = 0
    benchmark_values = []
    i = 0
    for d in dates_all:
        if benchmark_commodity:
            benchmark_current = benchmark_current * (1.0+benchmark_df.loc[pandas.to_datetime(d), "daily_returns"])
        
        benchmark_current = benchmark_current*target_daily_return

        while i < len(flows) and flows[i].date <= d:
            if flows[i].source != 'simulated-close':
                benchmark_current += -float(convert.convert_amount(flows[i].amount, target_currency, price_map, date=flows[i].date).number)
            i+=1

        benchmark_values.append(benchmark_current)
    return dates_all, benchmark_values

def get_amortized_value_plot_data_from_flows(price_map, flows, returns_rate, target_currency, dates):
    date_min = dates[0] - datetime.timedelta(days=1)
    date_max = dates[-1]
    num_days = (date_max - date_min).days
    dates_all = [dates[0] + datetime.timedelta(days=x) for x in range(num_days)]
    gamounts = np.zeros(num_days)
    rate = (1 + returns_rate) ** (1./365)
    for flow in flows:
        remaining_days = (date_max - flow.date).days
        if target_currency:
            amt = -float(convert.convert_amount(flow.amount, target_currency, price_map, date=flow.date).number)
        else:
            amt = -float(flow.amount.number)
        if remaining_days > 0:
            gflow = amt * (rate ** np.arange(0, remaining_days))
            gamounts[-remaining_days:] += gflow
        else:
            if flow.source != 'simulated-close':
                gamounts[-1] += amt
    return dates_all, gamounts

def plot_flows(output_dir: str,
               price_map: prices.PriceMap,
               flows: List[CashFlow],
               transactions: data.Entries,
               returns_rate: float,
               target_currency: Optional[Currency] = None) -> Dict[str, str]:
    """Produce plots from cash flows and returns, and more."""

    # Render cash flows.
    outplots = {}

    show_matplotlib = st.sidebar.checkbox('Show Matplotlib', False)
    if show_matplotlib:
        dates = [f.date for f in flows]
        dates_exdiv = [f.date for f in flows if not f.is_dividend]
        dates_div = [f.date for f in flows if f.is_dividend]
        #amounts = np.array([f.amount.number for f in flows])
        amounts_exdiv = np.array([f.amount.number for f in flows if not f.is_dividend])
        amounts_div = np.array([f.amount.number for f in flows if f.is_dividend])

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=[10, 4],
                                gridspec_kw={'height_ratios': [3, 1]})
        for ax in axs:
            set_axis(ax, dates[0] if dates else None, dates[-1] if dates else None)
            ax.axhline(0, color='#000', linewidth=0.2)
            ax.vlines(dates_exdiv, 0, amounts_exdiv, linewidth=3, color='#000', alpha=0.7)
            ax.vlines(dates_div, 0, amounts_div, linewidth=3, color='#0A0', alpha=0.7)
        axs[1].set_yscale('symlog')

        axs[0].set_title("Cash Flows")
        axs[1].set_title("log(Cash Flows)")
        fig.autofmt_xdate()
        fig.tight_layout()

        st.write(fig)

    df = investments.cash_flows_to_table(flows)
    log_plot = st.sidebar.checkbox('Log Plot', True)
    df['log'] = [np.log(amount) if amount > 0 else -np.log(np.abs(amount)) for amount in df['amount'] ]
    data = 'log' if log_plot else 'amount'

    fig = px.bar(df, x='date', y=data, barmode='overlay', color='is_dividend', labels={
                     "date": "Date",
                     "amount": "Cash Flow Amount ($)",
                     "log": "Cash Flow Amount ($)",
                     "is_dividend": "Dividend"
                 })
    fig.update_traces(width=1e9)

    if log_plot:
        vals = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
        pos_vals = [val for val in np.log(vals) if val < df['log'].max()]
        neg_vals = [val for val in -np.log(vals) if val > df['log'].min()]
        neg_vals.extend(pos_vals)
        text = [np.exp(val) if val > 0 else -np.exp(np.abs(val)) for val in neg_vals]
        st.write(text)
        text = [f'${val:.0f}' for val in text]
        fig.update_yaxes(
            ticktext=text,
            tickvals=neg_vals
        )
    st.plotly_chart(fig)
    st.write(df)

    # # Render cumulative cash flows, with returns growth.
    # lw = 0.8
    # if dates:
    #     dates_all, gamounts = get_amortized_value_plot_data_from_flows(price_map, flows, returns_rate, target_currency, dates)

    #     fig, ax = plt.subplots(figsize=[10, 4])
    #     ax.set_title("Cumulative value")
    #     set_axis(ax, dates[0] if dates else None, dates[-1] if dates else None)
    #     ax.axhline(0, color='#000', linewidth=lw)

    #     #ax.scatter(dates_all, gamounts, color='#000', alpha=0.2, s=1.0)
    #     ax.plot(dates_all, gamounts, color='#000', alpha=0.7, linewidth=lw)

    # # Overlay value of assets over time.
    # value_dates, value_values = returnslib.compute_portfolio_values(price_map, transactions, target_currency)
    # ax.plot(value_dates, value_values, color='#00F', alpha=0.5, linewidth=lw)
    # ax.scatter(value_dates, value_values, color='#00F', alpha=lw, s=2)

    # ax.legend(["Amortized value from flows", "Market value"], fontsize="xx-small")
    # fig.autofmt_xdate()
    # fig.tight_layout()
    # filename = path.join(output_dir, "cumvalue.svg")
    # outplots["cumvalue"] = "cumvalue.svg"
    # plt.savefig(filename)
    # plt.close(fig)

    return outplots



def write_price_directives(filename: str,
                           pricer: Pricer,
                           days_price_threshold: int):
    """Write a list of required price directives as a Beancount file."""
    price_entries = []
    for (currency, required_date), found_dates in sorted(pricer.required_prices.items()):
        assert len(found_dates) == 1
        cost_currency, actual_date, rate = found_dates.pop()
        days_late = (required_date - actual_date).days
        if days_late < days_price_threshold:
            continue
        price = data.Price({}, required_date, currency, Amount(rate, cost_currency))
        price_entries.append(price)
    os.makedirs(path.dirname(filename), exist_ok=True)
    with open(filename, "w") as prfile:
        printer.print_entries(price_entries, file=prfile)


def get_calendar_intervals(date: Date) -> List[Interval]:
    """Return a list of date pairs for sequential intervals."""
    intervals = [
        (str(year), Date(year, 1, 1), Date(year + 1, 1, 1))
        for year in range(TODAY.year - 15, TODAY.year)]
    intervals.append(
        (str(TODAY.year), Date(TODAY.year, 1, 1), date))
    return intervals


def get_cumulative_intervals(date: Date) -> List[Interval]:
    """Return a list of date pairs for sequential intervals."""
    return [
        ("15_years_ago", Date(date.year - 15, 1, 1), date),
        ("10_years_ago", Date(date.year - 10, 1, 1), date),
        ("5_years_ago", Date(date.year - 5, 1, 1), date),
        ("4_years_ago", Date(date.year - 4, 1, 1), date),
        ("3_years_ago", Date(date.year - 3, 1, 1), date),
        ("2_years_ago", Date(date.year - 2, 1, 1), date),
        ("1_year_ago", Date(date.year - 1, 1, 1), date),
        ("ytd", Date(date.year, 1, 1), date),
        ("rolling_6_months_ago", date - relativedelta(months=6), date),
        ("rolling_3_months_ago", date - relativedelta(months=3), date),
    ]


def generate_reports(account_data_map: Dict[Account, AccountData],
                     config: Config,
                     price_map: prices.PriceMap,
                     end_date: Date,
                     output_dir: str,
                     parallel: bool,
                     pdf: bool) -> Pricer:
    """Produce the list of requested reports."""

    pricer = Pricer(price_map)

    # Write out a returns file for every account.
    os.makedirs(output_dir, exist_ok=True)
    # multiprocessing.set_start_method(
    #     'spawn' if platform.system() == 'Windows' else 'fork')
    calls = []
    report = st.sidebar.selectbox('Group', config.groups.group, format_func=lambda x: x.name)
    
    adlist = [account_data_map[name] for name in report.investment]
    assert isinstance(adlist, list)
    assert all(isinstance(ad, AccountData) for ad in adlist)

    function = write_returns_st
    basename = path.join(output_dir, report.name)
    filename = "{}.pdf".format(basename) if pdf else basename
    calls.append(partial(
        function, filename,
        pricer, adlist, report.name,
        end_date,
        report.currency))

    calls.append(partial(
        write_returns_debugfile, "{}.org".format(basename),
        pricer, adlist, report.name,
        end_date,
        report.currency))

    if parallel:
        with multiprocessing.Pool(5) as pool:
            asyns = []
            for func in calls:
                asyns.append(pool.apply_async(func))
            for asyn in asyns:
                asyn.wait()
                assert asyn.successful()
    else:
        for func in calls:
            func()

    return pricer


def generate_report_mapper(item: Tuple[Group, List[AccountData]],
                           price_map: prices.PriceMap,
                           end_date: Date) -> Tuple[str, bytes]:
    """A mapper function that can be run from Beam to produce a PDF's bytes."""
    report, adlist = item
    with tempfile.NamedTemporaryFile("wb") as tmpfile:
        pricer = Pricer(price_map)
        write_returns_pdf(
            tmpfile.name, pricer, adlist, report.name, end_date, report.currency)
        tmpfile.flush()
        with open(tmpfile.name, "rb") as rfile:
            return (report.name, rfile.read())


def generate_price_pages(account_data_map: Dict[Account, AccountData],
                         price_map: prices.PriceMap,
                         output_dir: str):
    """Produce renders of price time series for each currency.
    This should help us debug issues with price recording, in particulawr,
    with respect to stock splits."""

    pricer = Pricer(price_map)

    # Write out a returns file for every account.
    os.makedirs(output_dir, exist_ok=True)
    pairs = set((ad.currency, ad.cost_currency)
                for ad in account_data_map.values()
                if ad.currency and ad.cost_currency)

    for base_quote in sorted(pairs):
        logging.info("Producing price page for %s", base_quote)
        all_prices = prices.get_all_prices(price_map, base_quote)
        if not all_prices:
            continue

        dates = np.array([date for date, _ in all_prices])
        prices_ = np.array([price for _, price in all_prices])

        fig, ax = plt.subplots(1, 1, figsize=[10, 4])
        set_axis(ax, dates[0], dates[-1])
        ax.set_title("Prices for {} ({})".format(*base_quote))
        ax.plot(dates, prices_, linewidth=0.5)
        ax.scatter(dates, prices_, s=2.0)
        fig.autofmt_xdate()
        fig.tight_layout()
        filename = path.join(output_dir, "{}_{}.svg".format(*base_quote))
        plt.savefig(filename)
        plt.close(fig)
