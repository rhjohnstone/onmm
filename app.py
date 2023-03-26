from ast import literal_eval

import lightning as L
import streamlit as st
from lightning.app.frontend import StreamlitFrontend
from lightning.app.structures import List

from mat_mul import MatMulFlow


class NoBuildConfig(L.BuildConfig):
    def on_work_init(self, work, cloud_compute=None):
        return


def transpose(x):
    return list(zip(*x))


def my_streamlit_ui(state):
    try:
        state.xss = literal_eval(st.text_input("Left matrix"))
    except:
        state.xss = None
    try:
        state.yss = transpose(literal_eval(st.text_input("Right matrix")))
    except:
        state.yss = None
    state.button = st.button("Compute matrix product")

    if state.result is not None:
        st.write(str(state.result))


class LitStreamlit(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.xss = None
        self.yss = None
        self.button = False
        self.result = None
        self.matmul_flows = List()
        self.active = False

    def run(self):
        if self.button and (self.xss is not None) and (self.yss is not None):
            self.button = False
            self.result = None
            self.matmul_flows.append(MatMulFlow(self.xss, self.yss))
            self.active = True
        if self.active:
            self.matmul_flows[-1].run()
            if self.matmul_flows[-1].result is not None:
                self.result = self.matmul_flows[-1].result
                self.active = False
                print(self.result)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=my_streamlit_ui)


app = L.LightningApp(LitStreamlit())
