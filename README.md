# Midterm Project Data Mining Course: Pablo Salar Carrera 
## Retail Transaction Analysis

## Description
This project involves analyzing retail transaction data from various stores, including Amazon, BestBuy, K-Mart, and Nike. The dataset contains transaction records along with associated item names, which are used for association rule mining and generating insights into customer behavior.

The project also includes scripts and notebooks to implement the analysis, as well as the results in HTML format.

## Folder Structure

The project is organized into the following directories:

- `/data/raw/`: Contains the raw transaction and item name data files for various retailers:
  - `Amazon_Transactions.csv` - Transaction data from Amazon.
  - `Amazon_itemNames.csv` - Item names corresponding to Amazon's transactions.
  - `BestBuy_Transactions.csv` - Transaction data from BestBuy.
  - `BestBuy_itemNames.csv` - Item names corresponding to BestBuy's transactions.
  - `K-mart_Transactions.csv` - Transaction data from K-mart.
  - `K-mart_itemNames.csv` - Item names corresponding to K-mart's transactions.
  - `Nike_Transactions.csv` - Transaction data from Nike.
  - `Nike_itemNames.csv` - Item names corresponding to Nike's transactions.
  - `Supermarket_Transactions.csv` - Transaction data from a Supermarket.
  - `Supermarket_itemNames.csv` - Item names corresponding to Supermarket transactions.
  - `LargeTransactionalData_Transactions.csv.zip` - A zipped file containing large-scale transaction data.
  - `LargeTransactionalData_itemNames.csv` - Item names for the large-scale transactional dataset.

- `/notebooks/`: Contains Jupyter notebooks for the analysis and visualization:
  - `RulesAssociation.ipynb` - Notebook for implementing association rule mining and analyzing results.

- `/src/`: Contains the Python scripts used for the analysis:
  - `RulesAssociation.py` - Python script for performing association rule mining.

- `/reports/`: Contains generated reports summarizing the results:
  - `RulesAssociation.html` - The HTML version of the final report summarizing the analysis and results.
  - `salar_pablo_midtermproj.pdf` - The PDF report for the midterm project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/psalarc/psalarc.git
