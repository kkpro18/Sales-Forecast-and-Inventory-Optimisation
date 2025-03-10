#!/bin/bash


AUTHOR="Kirisan Thiruchelvam" 
VERSION="1.0.0" 
PROJECT_DIR="/Users/kirisanthiruchelvam/PycharmProjects/Sales-Forecast-and-Inventory-Optimisation"
VENV_DIR="$PROJECT_DIR/.venv"

DATE="March 2025" 

# Simple header 
echo -e "${GREEN}==========================================" 
echo -e "${YELLOW} Mini 
Setup Script - Version $VERSION" 
echo -e "  Author: $AUTHOR" 
echo -e "  Date: $DATE" 
echo -e "${GREEN}==========================================" 
echo -e "${NC}" # Short description
echo -e "${YELLOW}This script allows you to easily run this application, setting up the Flask Application and the Web Application.${NC}"


# Opening Flask - Preparing for API Requests from Streamlit Application
echo "Setting up Flask App"
osascript -e "tell application \"Terminal\" to do script \"cd $PROJECT_DIR && source $VENV_DIR/bin/activate && python App/utils/flask_app.py\""

# Open a new Terminal window and run Streamlit app
osascript -e "tell application \"Terminal\" to do script \"cd $PROJECT_DIR && source $VENV_DIR/bin/activate && python -m streamlit run App/0_Home.py\""




