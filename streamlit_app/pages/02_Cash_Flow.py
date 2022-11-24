import pandas as pd
import streamlit as st
from beancount.core import data, prices

from beangrow import investments, reports, streamlit_helpers
from beangrow import returns as returnslib
from beangrow.returns import Pricer, Returns


def main():

    st.write('# Cash Flow')

    if 'args' not in st.session_state:
        st.write('### Run Main Page First')
        return

    """Write out returns report to a directory with files in it."""

    # TOOD(blais): Prices should be plot separately, by currency.
    # fprint("<h2>Prices</h2>")
    # pairs = set((r.currency, r.cost_currency) for r in account_data)
    # plots = plot_prices(dirname, pricer.price_map, pairs)
    # for _, filename in sorted(plots.items()):
    #     fprint('<img src={} style="width: 100%"/>'.format(filename))

    report = streamlit_helpers.select_report()

    if 'cash_flows' not in st.session_state:
        streamlit_helpers.load_report(report)
    
    # Render cash flows.
    show_pyplot = st.sidebar.checkbox('Show pyplot plot', False)
    if show_pyplot:
        fig = reports.plot_flows_pyplot(st.session_state.cash_flows)
        st.write(fig)

    log_plot = st.sidebar.checkbox('Log Plot', True)
    df = investments.cash_flows_to_table(st.session_state.cash_flows)
    fig = reports.plot_flows_plotly(df, log_plot)
    st.plotly_chart(fig)
    st.write(df)


if __name__ == '__main__':
    main()
