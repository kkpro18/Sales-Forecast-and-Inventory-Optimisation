import simpy
import streamlit as st

from utils.inventory_simulator import InventorySimulator, run_store, reset_simulation, total_overstocking_cost, \
    total_understocking_cost, total_restock_cost
from utils.session_manager import SessionManager

st.set_page_config(
    page_title="Inventory Policy Simulator",
    page_icon="⚙",
    layout="wide",
)

st.markdown("# Inventory Policy Simulator")
st.write("""Let's Begin Simulating Inventory Policies!""")

if not SessionManager.is_there("inventory_simulation_logs"):
    SessionManager.set_state("inventory_simulation_logs", "")

def input_details():

    initial_stock = int(st.slider("The initial stock level: ", 10, 500, 25))
    reorder_point = int(st.slider("The reorder point: ", 5, 350, 25))
    max_inventory_level = int(st.slider("The max Inventory level: ", 20,650, 25))
    overstocking_cost = float(st.slider("The overstock cost per unit: ", 1, 350, 1))
    understocking_cost = float(st.slider("The understock cost per unit: ",1, 350, 10))
    restock_cost = float(st.slider("The restock cost per order: ", 1,600,10))
    demand_per_period = float(st.slider("The demand per period e.g 2 per day: ", 1, 100, 1))

    return {
        'initial_stock': initial_stock,
        'reorder_point': reorder_point,
        'max_inventory_level': max_inventory_level,
        'overstocking_cost': overstocking_cost,
        'understocking_cost': understocking_cost,
        'restock_cost': restock_cost,
        'demand_per_period': demand_per_period
    }

env = simpy.Environment()

inventorySimulator = InventorySimulator(
    env,
    **input_details()
)
if st.button("Begin Simulation"):

    env.process(run_store(env, inventorySimulator))
    env.run(until=30)
    log_area = st.text_area("Inventory Simulation Log", value=SessionManager.get_state("inventory_simulation_logs"), height=300, max_chars=None, key="logs", disabled=True)

    st.write(f"\nTotal Overstocking Cost: {total_overstocking_cost}")
    st.write(f"\nTotal Understocking Cost: {total_understocking_cost}")
    st.write(f"Total Restock Cost: {total_restock_cost}")
    st.write(f"Total Cost: {total_overstocking_cost + total_understocking_cost + total_restock_cost}")
    st.write("Simulation Complete: Please check logs above for details.")

if st.button("Reset Simulation"):
    reset_simulation()
    st.success("Simulation Setup Reset Successfully")


