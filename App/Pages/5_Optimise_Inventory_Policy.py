import numpy as np
import streamlit as st
import App.utils.session_manager
from App.Models.inventory_simulator_model import InventorySimulator, run_store, reset_simulation, \
    total_overstocking_cost, total_understocking_cost, total_restock_cost
from App.Controllers import inventory_policy_simulator_controller

st.set_page_config(
    page_title="Calculate Optimal Inventory Policy",
    page_icon="⚙",
    layout="wide",
)

st.markdown("# Optimise Inventory Policy")
st.write("""Let's Begin Optimising Inventory Policies!""")


def eoq_input_details():
    st.write("Please Submit Details about Current Product Inventory Order Details (Holding Cost and Demand share same period e.g Monthly)")
    periodic_demand = int(st.slider("Expected Units To Be Sold Per Period X: ", 10, 10000, 25))
    order_cost = int(st.slider("Ordering Cost Per Order: ", 5, 20000, 25))
    holding_cost = int(st.slider("Holding Cost Per Unit Per Period X: ", 20, 5000, 25))

    if st.button("Submit Details"):
        return {
            'periodic_demand': periodic_demand,
            'order_cost': order_cost,
            'holding_cost': holding_cost,
        }
    return None

def calculate_eoq(periodic_demand, order_cost, holding_cost):
    return np.sqrt((2 * periodic_demand * order_cost) / holding_cost)

eoq_input = eoq_input_details()
if eoq_input is not None:
    st.write("The Optimal Order Quantity for this Product is: ", round(calculate_eoq(**eoq_input)))


st.page_link("pages/5_Optimise_Inventory_Policy.py", label="👈 Next Stage: Calculate Optimal inventory policy",
             icon="⚙️")

