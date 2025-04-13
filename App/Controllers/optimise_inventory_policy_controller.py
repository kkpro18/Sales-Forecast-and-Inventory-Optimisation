from App.Models import  optimise_inventory_policy_model
import streamlit as st

def handle_eoq_input():
    eoq_input = None
    try:
        eoq_input = optimise_inventory_policy_model.eoq_input_details()
    except Exception as e:
        st.error(e)
    finally:
        if eoq_input is not None:
            return eoq_input

def handle_eoq_calculation(periodic_demand, order_cost, holding_cost):
    optimal_order_quantity = None
    try:
        optimal_order_quantity = optimise_inventory_policy_model.calculate_eoq(periodic_demand, order_cost, holding_cost)
    except Exception as e:
        st.error(e)
    finally:
        if optimal_order_quantity is not None:
            return optimal_order_quantity


def handle_newsvendor_input():
    newsvendor_input = None
    try:
        newsvendor_input = optimise_inventory_policy_model.newsvendor_input_details()
    except Exception as e:
        st.error(e)
    finally:
        if newsvendor_input is not None:
            return newsvendor_input


def handle_newsvendor_calculation(average_periodic_demand, std_deviation_demand, understocking_cost, overstocking_cost):
    optimal_order_quantity = None
    try:
        optimal_order_quantity = optimise_inventory_policy_model.calculate_newsvendor(average_periodic_demand,
                                                                                      std_deviation_demand,
                                                                                      understocking_cost,
                                                                                      overstocking_cost)
    except Exception as e:
        st.error(e)
    finally:
        if optimal_order_quantity is not None:
            return optimal_order_quantity

