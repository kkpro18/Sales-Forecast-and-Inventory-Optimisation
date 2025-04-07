import simpy
import streamlit as st
from utils.session_manager import SessionManager
from App.Models.inventory_simulator_model import InventorySimulator, run_store, reset_simulation, total_overstocking_cost, total_understocking_cost, total_restock_cost
from App.Controllers import inventory_policy_simulator_controller

st.set_page_config(
    page_title="Inventory Policy Simulator",
    page_icon="⚙",
    layout="wide",
)

st.markdown("# Inventory Policy Simulator")
st.write("""Let's Begin Simulating Inventory Policies!""")

if not SessionManager.is_there("inventory_simulation_logs"):
    SessionManager.set_state("inventory_simulation_logs", "")

input_details = inventory_policy_simulator_controller.handle_input_details()

env = simpy.Environment()
inventorySimulator = InventorySimulator(env, **input_details)

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


