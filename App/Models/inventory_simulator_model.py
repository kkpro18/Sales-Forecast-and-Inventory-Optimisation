# Methods or Back End Code goes here for the Inventory Simulator

import random
import time

import simpy
import streamlit as st

from utils.session_manager import SessionManager

total_overstocking_cost = 0
total_understocking_cost = 0
total_restock_cost = 0

def input_details():
    initial_stock = int(st.slider("The initial stock level: ", 10, 500, 25))
    reorder_point = int(st.slider("The reorder point: ", 5, 350, 25))
    max_inventory_level = int(st.slider("The max Inventory level: ", 20, 650, 25))
    overstocking_cost = float(st.slider("The overstock cost per unit: ", 1, 350, 1))
    understocking_cost = float(st.slider("The understock cost per unit: ", 1, 350, 10))
    restock_cost = float(st.slider("The restock cost per order: ", 1, 600, 10))
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

def reset_simulation():
    global total_overstocking_cost, total_understocking_cost, total_restock_cost

    SessionManager.set_state("inventory_simulation_logs", "")
    SessionManager.set_state("reset_simulation", False)
    total_overstocking_cost = 0
    total_understocking_cost = 0
    total_restock_cost = 0

def update_log(message):
    SessionManager.set_state("inventory_simulation_logs", (SessionManager.get_state("inventory_simulation_logs")+f"{message}\n"))



class InventorySimulator(object):
    def __init__(self, env, initial_stock, reorder_point, max_inventory_level, overstocking_cost, understocking_cost,
                 restock_cost, demand_per_period):
        self.stock = initial_stock
        self.reorder_point = reorder_point
        self.max_inventory_level = max_inventory_level
        self.overstocking_cost = overstocking_cost
        self.understocking_cost = understocking_cost
        self.restock_cost = restock_cost
        self.demand_per_period = demand_per_period

        self.env = env
        self.inventory = simpy.Resource(env, capacity=1)

    def restock(self):
        global total_restock_cost

        order_quantity = self.demand_per_period
        update_log(f"Restocking Quantity {order_quantity}")

        yield self.env.timeout(2)
        self.stock += order_quantity
        total_restock_cost += self.restock_cost
        update_log(f"Restocked! New stock level: {self.stock}")

    def place_order(self):
        global total_understocking_cost

        if self.stock > 0:
            self.stock -= 1
            update_log(f"An order is placed, Stock Remaining {self.stock}")
        else:
            total_understocking_cost += self.understocking_cost
            update_log(f"Stock is missing, order can not be fulfilled, stockout")
            yield self.env.process(self.restock())

    def inventory_triggers(self):
        global total_overstocking_cost

        if self.stock < self.reorder_point:
            update_log(f"Reorder Point reached, restocking now")
            yield self.env.process(self.restock())

        if self.stock > self.max_inventory_level:
            overstocked_units = self.stock - self.max_inventory_level
            overstock_cost = self.overstocking_cost * overstocked_units
            total_overstocking_cost += overstock_cost
            update_log(f"Overstocked {overstocked_units} units, incurring costs of {overstock_cost}")


def customer_purchase(env, inventorySimulator):
    with inventorySimulator.inventory.request() as request:
        yield request
        yield env.process(inventorySimulator.place_order())
    yield env.process(inventorySimulator.inventory_triggers())

def run_store(env, inventorySimulator):
    env.process(customer_purchase(env, inventorySimulator))

    while True:
        yield env.timeout(random.randint(1, 10))
        env.process(customer_purchase(env, inventorySimulator))

