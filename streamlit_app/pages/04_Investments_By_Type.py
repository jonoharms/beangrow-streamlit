import streamlit as st

from beangrow import investments


def main():
    st.write('# Investments by Type')

    if 'args' not in st.session_state:
        st.write('### Run Main Page First')
        return

    account_data_map = st.session_state.account_data_map
    dcontext = st.session_state.options_map['dcontext']

    investments.write_transactions_by_type(account_data_map.values(), dcontext)


if __name__ == '__main__':
    main()
