#!/bin/bash


AUTHOR="Kirisan Thiruchelvam" 
VERSION="1.0.0" 

DATE="March 2025" 

# Simple header 
echo -e "${GREEN}==========================================" 
echo -e "${YELLOW}
Setup Script - Version $VERSION" 
echo -e "  Author: $AUTHOR" 
echo -e "  Date: $DATE" 
echo -e "${GREEN}==========================================" 
echo -e "${NC}" # Short description
echo -e "${YELLOW}This script installs all the necessary packages for the application.${NC}"


# install requirements
echo "INSTALLING REQUIREMENTS..."
pip install -r requirements.txt