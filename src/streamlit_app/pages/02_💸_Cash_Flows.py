import pandas as pd
import streamlit as st
from beancount.core import data, prices
from streamlit_extras.dataframe_explorer import dataframe_explorer

from beangrow import investments, reports
from beangrow import returns as returnslib
from beangrow import reportdata
from beangrow.returns import Pricer, Returns


def main():

    st.write('# Cash Flow')

    if 'ledger' not in st.session_state:
        st.write('## Run Returns First')
        return
    else:
        ledger = st.session_state.ledger

    """Explore the Cash Flows for the selected group."""

    # TOOD(blais): Prices should be plot separately, by currency.
    # fprint("<h2>Prices</h2>")
    # pairs = set((r.currency, r.cost_currency) for r in account_data)
    # plots = plot_prices(dirname, pricer.price_map, pairs)
    # for _, filename in sorted(plots.items()):
    #     fprint('<img src={} style="width: 100%"/>'.format(filename))

    report = ledger.select_report()
    if 'reportdata' not in st.session_state:
        reportdata = ledger.load_report(report)
        st.session_state['reportdata'] = reportdata
    else:
        reportdata = st.session_state.reportdata

    st.sidebar.markdown('---')
    st.sidebar.markdown('### Options')
    log_plot = st.sidebar.checkbox('Log Plot', False)
    show_pyplot = st.sidebar.checkbox('Show pyplot plot', False)

    df = investments.cash_flows_to_table(reportdata.cash_flows)

    df_new = dataframe_explorer(df)
    with st.expander('Show DataFrame'):
        st.write(df_new)

    fig = reports.plot_flows_plotly(df_new, log_plot)
    st.plotly_chart(fig, use_container_width=True)

    if show_pyplot:
        fig = reports.plot_flows_pyplot(reportdata.cash_flows)
        st.write(fig)


if __name__ == '__main__':
    main()
