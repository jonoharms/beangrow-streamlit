
from beancount.core import prices
from beangrow import reports
import streamlit as st
from beangrow.returns import Pricer, Returns
from beangrow import returns as returnslib
from beangrow import investments
from beangrow import reports
from beancount.core import data

def main():

    st.write("# Cash Flow")

    if 'args' not in st.session_state:
        st.write('### Run Main Page First')
        return

    args = st.session_state.args
    
    # Generate output reports.
    entries = st.session_state.entries
    account_data_map = st.session_state.account_data_map
    config = st.session_state.config
    end_date = st.session_state.end_date

    price_map = prices.build_price_map(entries)
    pricer = Pricer(price_map)
    report = st.sidebar.selectbox('Group', config.groups.group, format_func=lambda x: x.name)
    account_data = [account_data_map[name] for name in report.investment]
    # reports.write_returns_st(pricer, adlist, report.name, end_date, report.currency)

    """Write out returns report to a directory with files in it."""
    target_currency = report.currency
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

if __name__ == '__main__':
    main()