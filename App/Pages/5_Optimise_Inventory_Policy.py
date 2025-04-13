import numpy as np
import streamlit as st
import App.utils.session_manager
from App.Controllers import optimise_inventory_policy_controller

st.set_page_config(
    page_title="Calculate Optimal Inventory Policy",
    page_icon="⚙",
    layout="wide",
)

st.markdown("# Optimise Inventory Policy")
st.write("""Let's Begin Optimising Inventory Policies!""")

perishable = st.toggle("Perishable Product")
product_name = st.text_input("Product Name")
if product_name:
    st.subheader(product_name)
if not perishable:
    eoq_input = optimise_inventory_policy_controller.handle_eoq_input()
    if eoq_input:
        eoq_optimal_order_quantity = optimise_inventory_policy_controller.handle_eoq_calculation(**eoq_input)
        st.write("The Optimal Order Quantity for this Product is: ", eoq_optimal_order_quantity)
elif perishable:
    newsvendor_input = optimise_inventory_policy_controller.handle_newsvendor_input()
    if newsvendor_input:
        newsvendor_optimal_order_quantity = optimise_inventory_policy_controller.handle_newsvendor_calculation(**newsvendor_input)
        st.write("The Optimal Order Quantity for this Product is: ", newsvendor_optimal_order_quantity)




st.page_link("pages/5_Optimise_Inventory_Policy.py", label="👈 Next Stage: Calculate Optimal inventory policy",
             icon="⚙️")

