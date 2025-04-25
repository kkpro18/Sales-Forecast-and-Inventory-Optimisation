# Sales Forecasting and Inventory Optimisation Project

## Datasets:
- [Option 1 : (synthetic_retail_data)](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)
    - Option 1 holds synthetic data that is designed with realistic characteristics between 2022 and 2024
- [Option 2 : (india_retail_sales_data)](https://www.kaggle.com/datasets/balusami/retail-inventory-optimization)
    - Option 2 holds sales data from a retail store in India from 2010

## Steps:
Develop Unit Test
1. Collect Data [x]
2. Analyse Data [x]
3. Process & Prepare Data [ ]
4. Implement Time Series Model [ ]
5. Test and Fine-tune model [ ]
6. Calculate Optimal Inventory Policies EOQ and NewsVendor [ ]
7. Integrate into an Application (User-Interface) [ ]
8. Add Macro-Economic Variables [ ]
9. Generate Insights [ ]

## Project Architecture

This project follows a Model-View-Controller (MVC) pattern:

- **Models**: Contains the business logic and data handling
- **Controllers**: Acts as intermediaries between models and views
- **Pages**: Represents the user interface
- **Utils**: Contains utility functions

### UML Diagram

A UML diagram of the project architecture is available in the `App` directory:

- `App/uml_diagram.puml`: PlantUML file containing the UML diagram
- `App/uml_diagram_readme.md`: README file explaining the UML diagram and how to view it
- `App/generate_uml_diagram.sh`: Shell script to generate a PNG version of the UML diagram

To generate a PNG version of the UML diagram, run:

```bash
chmod +x App/generate_uml_diagram.sh
./App/generate_uml_diagram.sh
```

This will create `App/uml_diagram.png` which you can view with any image viewer.

For more details about the project architecture, please refer to `App/uml_diagram_readme.md`.
