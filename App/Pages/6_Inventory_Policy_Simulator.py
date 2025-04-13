import simpy
import streamlit as st
from App.utils.session_manager import SessionManager
from App.Models import inventory_simulator_model
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
inventorySimulator = inventory_simulator_model.InventorySimulator(env, **input_details)

if st.button("Begin Simulation"):
    env.process(inventory_simulator_model.run_store(env, inventorySimulator))
    env.run(until=30)
    log_area = st.text_area("Inventory Simulation Log", value=SessionManager.get_state("inventory_simulation_logs"), height=300, max_chars=None, key="logs", disabled=True)

    st.write(f"\nTotal Overstocking Cost: {inventory_simulator_model.total_overstocking_cost}")
    st.write(f"\nTotal Understocking Cost: {inventory_simulator_model.total_understocking_cost}")
    st.write(f"Total Restock Cost: {inventory_simulator_model.total_restock_cost}")
    st.write(f"Total Cost: {inventory_simulator_model.total_overstocking_cost + inventory_simulator_model.total_understocking_cost + inventory_simulator_model.total_restock_cost}")
    st.write("Simulation Complete: Please check logs above for details.")

if st.button("Reset Simulation"):
    inventory_simulator_model.reset_simulation()
    st.success("Simulation Setup Reset Successfully")

st.page_link("pages/7_Share_Feedback.py", label="👈 What do you think of this applicaiton?",
             icon="⚙️")
