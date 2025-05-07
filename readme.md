# Sales Forecasting and Inventory Optimisation Project

## How to Run Application

- **Installing Requirements**:
- Conda Environment Python 3.10
  - If running on a Mac, run the `install packages` shell script.
  - On other systems, run `pip install -r requirements.txt` in the terminal.

- **Running Application**: 
  - On a Mac, run the `run_app` shell script.
  - On other systems:
    1. In one terminal, run:
       ```bash
       uvicorn App.utils.fastapi.main:app --port 8000 --reload
       ```
    2. In another terminal, run:
       ```bash
       python -m streamlit run App/0_Home.py
       ```
  
- Enjoy!

---

## Project Architecture

This project follows a **Model-View-Controller (MVC)** pattern:

- **Models**: Contains the business logic and data handling.
- **Pages**: Represents the user interface.
- **Controllers**: Acts as intermediaries between models and views.
- **Utils**: Contains utility functions.
