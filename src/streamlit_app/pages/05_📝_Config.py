import streamlit as st

def main():
    st.write('# Config')

    if 'ledger' not in st.session_state:
        st.write('### Run Main Page First')
        return
    
    col0, col1 = st.columns(2)

    with col0:
        st.write('## Investments')
        with st.expander('Show/Hide', expanded=True):
            st.text(st.session_state.ledger.config.investments)

    with col1:
        st.write('## Groups')
        with st.expander('Show/Hide', expanded=True):
            st.text(st.session_state.ledger.config.groups)


if __name__ == '__main__':
    main()
