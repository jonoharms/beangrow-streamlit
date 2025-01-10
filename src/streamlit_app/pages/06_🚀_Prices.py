import streamlit as st


def main():
    st.write('# Commodity Prices')

    if 'ledger' not in st.session_state:
        st.write('### Run Main Page First')
        return

    ledger = st.session_state.ledger
    fig = ledger.generate_price_pages()
    st.write(fig)


if __name__ == '__main__':
    main()
