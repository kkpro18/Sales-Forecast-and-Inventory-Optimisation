#!/bin/bash


AUTHOR="Kirisan Thiruchelvam" 
VERSION="1.1"
PROJECT_DIR="/Users/kirisanthiruchelvam/PycharmProjects/Sales-Forecast-and-Inventory-Optimisation"
VENV_DIR="$PROJECT_DIR/.venv"

DATE="April 2025"

# Simple header 
echo -e "${GREEN}==========================================" 
echo -e "${YELLOW} Mini 
Setup Script - Version $VERSION" 
echo -e "  Author: $AUTHOR" 
echo -e "  Date: $DATE" 
echo -e "${GREEN}==========================================" 
echo -e "${NC}" # Short description
echo -e "${YELLOW}This script allows you to easily run this application, setting up the FastAPI Application.${NC}"


# Opening FastAPI - Preparing for API Requests from Streamlit Application
echo "Setting up FastAPI for incoming requests..."
osascript -e "tell application \"Terminal\" to do script \"cd $PROJECT_DIR && source $VENV_DIR/bin/activate && uvicorn App.utils.fastapi.main:app --port 8000 --reload\""


