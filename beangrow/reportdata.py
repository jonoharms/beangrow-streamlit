import datetime
import logging

import pandas as pd
import streamlit as st
from beancount import loader
from beancount.core import data, getters, prices
import plotly.express as px
from beangrow import config as configlib
from beangrow import investments, reports
from beangrow import returns as returnslib
from beangrow.returns import Pricer, Returns
from attrs import define
from typing import NamedTuple
from beangrow.config_pb2 import (
    Group,
) 

Date = datetime.date
TODAY = Date.today()


@define
class ReportData:
    report: Group
    cash_flows: pd.DataFrame
    returns: Returns
    cumulative_returns: list[Returns]
    calendar_returns: list[Returns]
    portfolio_value: pd.DataFrame
    transactions: data.Entries
    accounts: pd.DataFrame

    def plot_plotly(self):
        fig = px.line(self.portfolio_value)
        fig.update_layout(hovermode='x unified')
        fig.update_layout(height=600)

        return fig
        

  