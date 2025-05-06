# Sales Forecasting and Inventory Optimisation Project

## How to Run Application
- **Installing Requirements**: If running on a Mac, run the install packages shell script; else type "pip install -r requirements.txt" 
- **Running Application**: If running on a Mac, run the run_app shell script, else in one bash terminals type "uvicorn App.utils.fastapi.main:app --port 8000 --reload" and in another type "python -m streamlit run App/0_Home.py"
- Enjoy!
- 
## Project Architecture

This project follows a Model-View-Controller (MVC) pattern:

- **Models**: Contains the business logic and data handling
- **Pages**: Represents the user interface
- - **Controllers**: Acts as intermediaries between models and views
- **Utils**: Contains utility functions


