# Pricing Optimization ML

A machine learning system that predicts optimal markup percentages for product pricing based on order characteristics and customer attributes.

## Overview

This project demonstrates how to build a pricing optimization model using historical order data. The model predicts the appropriate markup percentage for each order based on:

- **Order features**: delivery distance, quantity, unit cost, packaging type
- **Customer features**: industry, credit limit, payment terms

## Dataset

The dataset contains ~12,800 orders and ~735 customers with the following key columns:

**Order Data (`order_df.csv`)**:
| Column | Description |
|--------|-------------|
| `delivery_distance` | Distance in miles |
| `order_quantity_lbs` | Order quantity in pounds |
| `average_unit_cost_lbs` | Unit cost in USD/lb |
| `packaging` | Packaging type (BULK, -MDRM) |
| `markup_percentage` | Target variable |
| `customer_id` | Link to customer data |

**Customer Data (`customer_df.csv`)**:
| Column | Description |
|--------|-------------|
| `industry_name` | Customer industry (Pharma, Food, etc.) |
| `credit_limit` | Customer credit limit |
| `term_code` | Payment terms |

## Models

| Model | Test R² | Notes |
|-------|---------|-------|
| Linear Regression | 0.63 | Baseline model |
| Decision Tree | 0.75 | Depth limited to 5 |
| Random Forest | 0.87 | 100 estimators, depth 10 |
| **XGBoost** | **0.88** | Best performance with customer features |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook pricing_optimization.ipynb
```

## Project Structure

```
├── pricing_optimization.ipynb  # Main analysis notebook
├── customer_df.csv             # Customer attributes
├── order_df.csv                # Historical orders
├── inference_df.csv            # New orders for prediction
├── requirements.txt            # Python dependencies
└── README.md
```

## Key Findings

1. **Customer industry matters**: Chemical Processing and Pharma customers typically receive different markup rates
2. **Order quantity impacts pricing**: Larger bulk orders tend to have different markup strategies
3. **Delivery distance affects costs**: Longer distances increase delivery costs which impacts optimal markup

## Usage

The trained model can be used to predict markup percentages for new orders:

```python
# Load the model and predict
import joblib
model = joblib.load('xgboost_model.pkl')
predicted_markup = model.predict(new_order_features)

# Calculate selling price
selling_price = unit_cost * (1 + predicted_markup)
```

## License

MIT License
