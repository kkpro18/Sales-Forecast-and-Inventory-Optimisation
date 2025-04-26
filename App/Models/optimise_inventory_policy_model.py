from scipy.stats import norm
import numpy as np
import streamlit as st

def eoq_input_details():
    st.write(
        "Please Submit Details about Current Product Inventory Order Details")
    periodic_demand = int(st.slider("Expected Units To Be Sold Per Period X: ", 1, 10000, 25))
    order_cost = int(st.slider("Ordering Cost Per Order: ", 1, 20000, 25))
    holding_cost = int(st.slider("Holding Cost Per Unit Per Period X: ", 1, 5000, 25))

    if st.button("Submit Input For EOQ Calculation"):
        return {
            'periodic_demand': periodic_demand,
            'order_cost': order_cost,
            'holding_cost': holding_cost,
        }
    return None


def calculate_eoq(periodic_demand, order_cost, holding_cost):
    return round(np.sqrt((2 * periodic_demand * order_cost) / holding_cost))

def newsvendor_input_details():
    st.write(
        "Please Submit Details about Current Product Inventory Order Details")
    average_periodic_demand = int(st.slider("Expected Average Units To Be Sold Per Period X: ", 1, 10000, 25))
    std_deviation_demand = int(st.slider("Standard Deviation of the Demand during Period X: ", 1, 5000, 25))
    understocking_cost = int(st.slider("Costs Faced from Understocking Per Unit: ", 1, 20000, 25))
    overstocking_cost = int(st.slider("Costs Faced from Overstocking Per Unit: ", 1, 5000, 25))

    if st.button("Submit Input For NewsVendor Calculation"):
        return {
            'average_periodic_demand': average_periodic_demand,
            "std_deviation_demand": std_deviation_demand,
            'understocking_cost': understocking_cost,
            'overstocking_cost': overstocking_cost,
        }
    return None


def calculate_newsvendor(average_periodic_demand, std_deviation_demand, understocking_cost, overstocking_cost):
    CR = understocking_cost / (understocking_cost + overstocking_cost)
    Z_Score = norm.ppf(CR)  # applies f^-1 by finding Z value, assumes normally distributed sales
    return round(average_periodic_demand + Z_Score * std_deviation_demand)
