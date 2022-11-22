
from beancount.core import prices
from beangrow import reports
import streamlit as st

def main():

    st.write("# Cash Flow")

    if 'args' not in st.session_state:
        st.write('### Run Main Page First')
        return

    args = st.session_state.args
    
    # Generate output reports.
    output_reports = args.output.joinpath("groups")
    entries = st.session_state.entries
    account_data_map = st.session_state.account_data_map
    config = st.session_state.config
    end_date = st.session_state.end_date


    st.text(f'Number of entries loaded: {len(entries)}')
    pricer = reports.generate_reports(account_data_map, config,
                                    prices.build_price_map(entries),
                                    end_date,
                                    output_reports,
                                    args.parallel, args.pdf)

if __name__ == '__main__':
    main()