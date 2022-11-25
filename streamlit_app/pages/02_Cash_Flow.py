import pandas as pd
import streamlit as st
from beancount.core import data, prices

from beangrow import investments, reports, streamlit_helpers
from beangrow import returns as returnslib
from beangrow.returns import Pricer, Returns
from streamlit_extras.dataframe_explorer import dataframe_explorer

def main():

    st.write('# Cash Flow')

    if 'args' not in st.session_state:
        st.write('### Run Main Page First')
        return

    """Explore the Cash Flows for the selected group."""

    # TOOD(blais): Prices should be plot separately, by currency.
    # fprint("<h2>Prices</h2>")
    # pairs = set((r.currency, r.cost_currency) for r in account_data)
    # plots = plot_prices(dirname, pricer.price_map, pairs)
    # for _, filename in sorted(plots.items()):
    #     fprint('<img src={} style="width: 100%"/>'.format(filename))

    report = streamlit_helpers.select_report()

    if 'cash_flows' not in st.session_state:
        streamlit_helpers.load_report(report)
    
    # # Render cash flows.
    # show_pyplot = st.sidebar.checkbox('Show pyplot plot', False)
    # if show_pyplot:
    #     fig = reports.plot_flows_pyplot(st.session_state.cash_flows)
    #     st.write(fig)

    df = investments.cash_flows_to_table(st.session_state.cash_flows)

    df_new = dataframe_explorer(df)
    with st.expander('Show DataFrame'):
        st.write(df_new)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Options")
    log_plot = st.sidebar.checkbox('Log Plot', False)
    fig = reports.plot_flows_plotly(df_new, log_plot)
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
