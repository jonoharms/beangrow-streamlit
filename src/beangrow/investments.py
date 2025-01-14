"""Library code to extract time-series data for each investment.

This code produces a data object for each investment from a Beancount ledger,
containing the list of transactions for its asset account, a list of cash flows
to compute the returns, and a bit more. Investments are defined as "one
commodity in one account" and can be further combined as reports in code outside
this library.
"""

__copyright__ = 'Copyright (C) 2020  Martin Blais'
__license__ = 'GNU GPLv2'

import collections
import datetime
import enum
import logging
import re
import sys
from typing import Optional, Tuple

import pandas
import streamlit as st
from attrs import define
from beancount.core import convert, data, display_context, getters
from beancount.core.amount import Amount
from beancount.core.inventory import Inventory
from beancount.parser import printer

from beangrow.config_pb2 import (  # pyright: reportGeneralTypeIssues=false
    Config,
    Investment,
    InvestmentConfig,
)

# Basic type aliases.
Account = str
Currency = str
Date = datetime.date


class Cat(enum.Enum):
    """Posting categorization.

    This is used to produce unique templates to categorize each transaction. A
    template is a set of the categories below, which each of a transaction's
    postings are classified with.
    """

    # The account holding the commodity.
    ASSET = 1

    # Cash accounts, employer matches, contributions, i.e., anything external to
    # the investment.
    CASH = 2

    # Dividend income account.
    DIVIDEND = 3

    # Commissions, fees and other expenses.
    EXPENSES = 4

    # Non-dividend income, P/L, gains, or other.
    INCOME = 5

    # Other assets than the primary asset for this investment.
    OTHERASSET = 7

    # Any other account.
    OTHER = 6


# Al list of dated cash flows. This is the unit that this program operates in,
# the sanitized time-series that allows us to compute returns.
@define
class CashFlow:
    date: Date
    amount: Amount
    is_dividend: bool
    source: str
    account: Account

    def to_dict(self):
        amount = float(self.amount.number) if self.amount.number else None
        d = {
            'date': self.date,
            'amount': amount,
            'currency': self.amount.currency,
            'is_dividend': self.is_dividend,
            'source': self.source,
            'investment': self.account,
        }

        return d


# All flow information associated with an account.
@define
class AccountData:
    account: Account
    currency: Currency
    cost_currency: Optional[Currency]
    commodity: data.Commodity
    open: data.Open
    close: data.Close
    cash_flows: list[CashFlow]
    transactions: data.Entries
    balance: Inventory
    catmap: dict[Account, Cat]


def categorize_accounts(
    config: InvestmentConfig, investment: Investment, accounts: set[Account]
) -> dict[Account, Cat]:
    """Categorize the type of accounts encountered for a particular investment's
    transactions. Our purpose is to make the types of postings generic, so they
    can be categorized and handled generically later on.
    """
    catmap = {}
    for account in accounts:
        if account == investment.asset_account:
            cat = Cat.ASSET
        elif account in investment.dividend_accounts:
            cat = Cat.DIVIDEND
        elif account in investment.cash_accounts:
            cat = Cat.CASH
        elif re.match(config.income_regexp or 'Income:', account):
            cat = Cat.INCOME
        elif re.match(config.expenses_regexp or 'Expenses:', account):
            cat = Cat.EXPENSES
        else:
            # Note: When applied, if the corresponding postings has a cost,
            # OTHERASSET will be assigned; otherwise OTHER will be assigned.
            cat = None
        catmap[account] = cat
    return catmap


def categorize_entry(
    catmap: dict[Account, Cat], entry: data.Directive
) -> Tuple[Cat]:
    """Assigns metadata to each posting."""
    postings = []
    for posting in entry.postings:
        category = catmap[posting.account]
        if category is None:
            category = Cat.OTHER if posting.cost is None else Cat.OTHERASSET
        meta = posting.meta.copy() if posting.meta else {}
        meta['category'] = category
        postings.append(posting._replace(meta=meta))
    return entry._replace(postings=postings)


def compute_transaction_signature(entry: data.Directive):
    """Compute a unique signature for each transaction."""
    categories = set(posting.meta['category'] for posting in entry.postings)
    sigtuple = tuple(sorted(categories, key=lambda item: item.value))
    return '_'.join(s.name for s in sigtuple)


_signature_registry = {}


def register(categories, description):
    """Registers a handler for a particular template/signature transaction."""

    def decorator(func):
        key = '_'.join(
            c.name for c in sorted(categories, key=lambda c: c.value)
        )
        _signature_registry[key] = (func, description)
        return func

    return decorator


def get_description(signature):
    _, description = _signature_registry.get(signature, (None, None))
    return description


def produce_cash_flows_general(
    entry: data.Directive, account: Account
) -> list[CashFlow]:
    """Produce cash flows using a generalized rule."""
    has_dividend = any(
        posting.meta['category'] == Cat.DIVIDEND for posting in entry.postings
    )
    flows = []
    for posting in entry.postings:
        category = posting.meta['category']
        if category == Cat.CASH:
            assert not posting.cost
            cf = CashFlow(
                entry.date,
                convert.get_weight(posting),
                has_dividend,
                'cash',
                account,
            )
            posting.meta['flow'] = cf
            flows.append(cf)

        elif category == Cat.OTHERASSET:
            # If the account deposits other assets, count this as an outflow.
            cf = CashFlow(
                entry.date,
                convert.get_weight(posting),
                False,
                'other',
                account,
            )
            posting.meta['flow'] = cf
            flows.append(cf)

    return flows


def produce_cash_flows_explicit(
    entry: data.Directive, account: Account
) -> list[CashFlow]:
    """Produce cash flows using explicit handlers from signatures."""
    sig = entry.meta['signature']
    try:
        handler, _ = _signature_registry[sig]
    except KeyError:
        epr = printer.EntryPrinter(stringify_invalid_types=True)
        print(epr(entry), file=sys.stderr)
        raise
    return handler(entry, account)


@register([Cat.ASSET], 'Stock splits, and conversions at same cost basis')
def handle_no_flows(*args) -> list[CashFlow]:
    """Exchanges of the same asset produce no cash flows."""
    return []


@register([Cat.ASSET, Cat.DIVIDEND], 'Asset dividend reinvested')
@register([Cat.ASSET, Cat.DIVIDEND, Cat.EXPENSES], 'Asset dividend reinvested')
def handle_dividend_reinvestments(*args) -> list[CashFlow]:
    """Reinvested stock dividends remains internal, the money is just moved to more
    of the asset. Note that because of this, it would make it difficult to
    remove the dividend from the performance of this asset."""
    return []


@register([Cat.ASSET, Cat.EXPENSES], 'Fee paid from liquidation')
def handle_fee_from_liquidation(*args) -> list[CashFlow]:
    """Fees paid purely from sales of assets. No in or out flows, the stock value is
    simply reduced."""
    return []


@register([Cat.ASSET, Cat.INCOME], 'Cost basis adjustment')
@register([Cat.ASSET, Cat.INCOME, Cat.OTHER], 'Cost basis adjustment')
@register(
    [Cat.ASSET, Cat.INCOME, Cat.EXPENSES], 'Fee from liquidation (with P/L)'
)
def handle_cost_basis_adjustments0(*args) -> list[CashFlow]:
    """No cash is disbursed for these adjustments, just a change in basis. This
    affects tax only. There are no associated cash flows."""
    return []


@register([Cat.EXPENSES, Cat.OTHER], 'Internal expense')
@register([Cat.OTHER], 'Movement between internal accounts')
def handle_cost_basis_adjustments1(*args) -> list[CashFlow]:
    """It's internal changes, no flows."""
    return []


@register([Cat.ASSET, Cat.CASH], 'Regular purchase or sale')
@register([Cat.ASSET, Cat.CASH, Cat.OTHER], 'Regular purchase or sale')
@register(
    [Cat.ASSET, Cat.CASH, Cat.INCOME], 'Regular purchase or sale, with P/L'
)
@register(
    [Cat.ASSET, Cat.CASH, Cat.INCOME, Cat.OTHER],
    'Regular purchase or sale, with P/L',
)
@register(
    [Cat.ASSET, Cat.CASH, Cat.EXPENSES, Cat.INCOME],
    'Regular purchase or sale, with expense and P/L',
)
@register(
    [Cat.ASSET, Cat.CASH, Cat.EXPENSES, Cat.OTHER], 'Regular purchase or sale'
)
@register(
    [Cat.ASSET, Cat.CASH, Cat.EXPENSES, Cat.INCOME, Cat.OTHER],
    'Regular purchase or sale',
)
@register(
    [Cat.ASSET, Cat.CASH, Cat.EXPENSES],
    'Regular purchase or sale, with expense',
)
def handle_buy_sell(entry: data.Directive, account: Account) -> list[CashFlow]:
    """In a regular purchase or sale, use the cash component for sales and purchases."""
    return _handle_cash(entry, account, False)


@register([Cat.CASH, Cat.DIVIDEND], 'Cash dividend')
@register([Cat.CASH, Cat.DIVIDEND, Cat.INCOME], 'Cash dividend')
@register(
    [Cat.CASH, Cat.EXPENSES, Cat.DIVIDEND], 'Cash dividend, with expenses'
)
def handle_dividends(
    entry: data.Directive, account: Account
) -> list[CashFlow]:
    """Dividends received in cash."""
    return _handle_cash(entry, account, True)


@register([Cat.CASH, Cat.EXPENSES], 'Cash for expense')
@register([Cat.CASH, Cat.OTHER], 'Cash for other internal account')
@register(
    [Cat.CASH, Cat.EXPENSES, Cat.OTHER], 'Cash for expense or something else'
)
@register([Cat.CASH, Cat.EXPENSES, Cat.INCOME], 'Recoveries from P2P lending')
def handle_cash_simple(
    entry: data.Directive, account: Account
) -> list[CashFlow]:
    """Cash for income."""
    return _handle_cash(entry, account, False)


def _handle_cash(
    entry: data.Directive, account: Account, is_dividend: bool
) -> list[CashFlow]:
    """In a regular purchase or sale, use the cash component for sales and purchases."""
    flows = []
    for posting in entry.postings:
        if posting.meta['category'] == Cat.CASH:
            assert not posting.cost
            cf = CashFlow(
                entry.date, posting.units, is_dividend, 'cash', account
            )
            posting.meta['flow'] = cf
            flows.append(cf)
    return flows


@register([Cat.ASSET, Cat.OTHERASSET], 'Exchange of assets')
def handle_stock_exchange(
    entry: data.Directive, account: Account
) -> list[CashFlow]:
    """This is for a stock exchange, similar to the issuance of GOOG from GOOGL."""
    flows = []
    for posting in entry.postings:
        if posting.meta['category'] == Cat.OTHERASSET:
            cf = CashFlow(
                entry.date,
                convert.get_weight(posting),
                False,
                'other',
                account,
            )
            posting.meta['flow'] = cf
            flows.append(cf)
    return flows


def extract_transactions_for_account(
    entries: data.Entries, config: Investment
) -> data.Entries:
    """Get the list of transactions affecting an investment account."""
    match_accounts = set([config.asset_account])
    match_accounts.update(config.dividend_accounts)
    match_accounts.update(config.match_accounts)
    return [
        entry
        for entry in data.filter_txns(entries)
        if any(posting.account in match_accounts for posting in entry.postings)
    ]


def process_account_entries(
    entries: data.Entries,
    config: InvestmentConfig,
    investment: Investment,
    check_explicit_flows: bool,
) -> Optional[AccountData]:
    """Process a single account."""
    account = investment.asset_account
    # logging.info('Processing account: %s', account)

    # Extract the relevant transactions.
    transactions = extract_transactions_for_account(entries, investment)
    if not transactions:
        logging.warning('No transactions for %s; skipping.', account)
        return None

    # Categorize the set of accounts encountered in the filtered transactions.
    seen_accounts = {
        posting.account for entry in transactions for posting in entry.postings
    }
    catmap = categorize_accounts(config, investment, seen_accounts)

    # Process each of the transactions, adding derived values as metadata.
    cash_flows = []
    balance = Inventory()
    decorated_transactions = []
    for entry in transactions:

        # Compute the signature of the transaction.
        entry = categorize_entry(catmap, entry)
        signature = compute_transaction_signature(entry)
        entry.meta['signature'] = signature

        # TODO(blais): Cache balance in every transaction to speed up
        # computation? Do this later.
        if False:
            # Update the total position in the asset we're interested in.
            for posting in entry.postings:
                if posting.meta['category'] is Cat.ASSET:
                    balance.add_position(posting)

        # Compute the cash flows associated with the transaction.
        flows_general = produce_cash_flows_general(entry, account)
        if check_explicit_flows:
            # Attempt the explicit method.
            flows_explicit = produce_cash_flows_explicit(entry, account)
            if flows_explicit != flows_general:
                print(
                    'Differences found between general and explicit methods:'
                )
                print('Explicit handlers:')
                for flow in flows_explicit:
                    print('  ', flow)
                print('General handler:')
                for flow in flows_general:
                    print('  ', flow)
                raise ValueError(
                    'Differences found between general and explicit methods:'
                )

        cash_flows.extend(flows_general)
        decorated_transactions.append(entry)

    cost_currencies = set(cf.amount.currency for cf in cash_flows)
    # assert len(cost_currencies) == 1, str(cost_currencies)
    cost_currency = cost_currencies.pop() if cost_currencies else None

    currency = investment.currency
    commodity_map = getters.get_commodity_directives(entries)
    comm = commodity_map[currency] if currency else None

    open_close_map = getters.get_account_open_close(entries)
    opn, cls = open_close_map[account]

    # Compute the final balance.
    balance = compute_balance_at(decorated_transactions)

    return AccountData(
        account,
        currency,
        cost_currency,
        comm,
        opn,
        cls,
        cash_flows,
        decorated_transactions,
        balance,
        catmap,
    )


def prune_entries(entries: data.Entries, config: Config) -> data.Entries:
    """Prune the list of entries to exclude all transactions that include a
    commodity name in at least one of its postings. This speeds up the
    recovery process by removing the majority of non-trading transactions."""

    commodities = set(
        aconfig.currency for aconfig in config.investments.investment
    )
    accounts = set()
    for aconfig in config.investments.investment:
        accounts.add(aconfig.asset_account)
        accounts.update(aconfig.dividend_accounts)
        accounts.update(aconfig.match_accounts)

    return [
        entry
        for entry in entries
        if (
            (
                isinstance(entry, data.Commodity)
                and entry.currency in commodities
            )
            or (isinstance(entry, data.Open) and entry.account in accounts)
            or (
                isinstance(entry, data.Transaction)
                and any(
                    posting.account in accounts for posting in entry.postings
                )
            )
        )
    ]


def compute_balance_at(
    transactions: data.Entries, date: Optional[Date] = None
) -> Inventory:
    """Compute the balance at a specific date."""
    balance = Inventory()
    for entry in transactions:
        if date is not None and entry.date >= date:
            break
        for posting in entry.postings:
            if posting.meta['category'] is Cat.ASSET:
                balance.add_position(posting)
    return balance


def write_account_file(
    dcontext: display_context.DisplayContext, account_data: AccountData
):
    """Write out a file with details, for inspection and debugging."""

    epr = printer.EntryPrinter(dcontext=dcontext, stringify_invalid_types=True)
    col1, col2 = st.columns(2)

    with col1:
        # Print front summary section.
        st.write('### Summary')

        # Print the final balance of the account.
        units_balance = account_data.balance.reduce(convert.get_units)
        st.write('Balance: {}'.format(units_balance))

        # Print out those details.
        st.write('### Category map')
        st.write(account_data.catmap)

        # Flatten cash flows to a table.
        st.write('### Cash flows\n')
        df = cash_flows_to_table(account_data.cash_flows)
        st.write(df)

    with col2:
        st.write('### Transactions\n')

        for entry in account_data.transactions:
            st.text(epr(entry))


def cash_flows_to_table(cash_flows: list[CashFlow]) -> pandas.DataFrame:
    """Flatten a list of cash flows to an HTML table string."""
    records = [flow.to_dict() for flow in cash_flows]
    df = pandas.DataFrame.from_records(records)

    return df


def write_transactions_by_type(
    account_data: list[AccountData], dcontext: display_context.DisplayContext
):
    """Write files of transactions by signature, for debugging."""

    # Build signature map.
    signature_map = collections.defaultdict(list)
    for accdata in account_data:
        for entry in accdata.transactions:
            signature_map[entry.meta['signature']].append(entry)

    # col1, ol2 = st.columns(2)

    summary = {
        sig: [get_description(sig), len(sigentries)]
        for sig, sigentries in signature_map.items()
    }
    df = pandas.DataFrame(
        index=list(summary.keys()),
        data=summary.values(),
        columns=['Description', 'Number of Entries'],
    )

    st.write('## Summary')
    st.dataframe(df, use_container_width=True)

    st.write('## Transactions')
    # Render them to files, for debugging.
    sig = st.sidebar.selectbox('Select Signature', signature_map.keys())
    sigentries = signature_map[sig]
    sigentries = data.sorted(sigentries)

    description = get_description(sig) or '?'
    st.write('**Description:** {}'.format(description))
    st.write('**Number of entries:** {}'.format(len(sigentries)))
    epr = printer.EntryPrinter(dcontext=dcontext, stringify_invalid_types=True)
    for entry in sigentries:
        st.text(epr(entry))


def extract(
    entries: data.Entries,
    config: Config,
    end_date: Date,
    check_explicit_flows: bool,
) -> dict[Account, AccountData]:
    """Extract data from the list of entries."""
    # Note: It might be useful to have an option for "the end of its history"
    # for Ledger that aren't updated up to today.

    # Remove all data after end date, if specified.
    if end_date < entries[-1].date:
        entries = [entry for entry in entries if entry.date < end_date]

    # Prune the list of entries for performance.
    pruned_entries = prune_entries(entries, config)

    # Process all the accounts.
    account_data = [
        process_account_entries(
            pruned_entries, config.investments, aconfig, check_explicit_flows
        )
        for aconfig in config.investments.investment
    ]
    account_data = list(filter(None, account_data))
    account_data_map = {ad.account: ad for ad in account_data}

    return account_data_map
