# Analytical Case Study - Olist

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a case study analyzing the Olist dataset from Kaggle. The project simulates an analysis project in a consulting firm, aiming to identify Olist's business challenges and propose actionable recommendations.  This analysis utilizes the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?resource=download) from Kaggle.

## Overview

This project provides a comprehensive analysis of Olist's business challenges, offering strategic recommendations based on data-driven insights. It mimics a consulting project workflow, covering data processing, analysis, and visualization. This analysis aims to uncover key trends and insights related to Olist's performance and identify areas for improvement.

## Project Goals

*   Identify key business challenges Olist faces.
*   Provide actionable recommendations to address these challenges.
*   Simulate a real-world data analysis project in a consulting context.

## Key Findings

*   **Growth Stagnation:** Since its launch, Olist experienced steady sales growth; however, growth has plateaued since 2018.
*   **Slowdown Across Key Categories:** Even within major product categories that contribute significantly to total sales, a slowdown in growth was observed, indicating an overall plateau in sales across various segments.
*   **Delivery Issues & Customer Complaints:** Review analysis revealed an increasing trend in customer complaints related to delivery issues, particularly highlighting "delivery delays."
*   **Delivery Delays: A Primary Challenge:** Delivery delays are suggested to be caused by extended lead times from logistics providers, and this issue appears to be a major business challenge, closely linked to Olist's growth stagnation.
*   **Lack of Clear Correlation:** Based on the available dataset, no clear correlation was identified between delivery delays and attributes related to buyers, sellers, or products.
*   **Further Investigation Needed:** To accurately identify the causes of delivery delays, additional interviews and detailed investigations into the operations of logistics providers and related data are required.

## Recommended Actions

*   Conduct a thorough investigation into the logistics processes and performance of providers.
*   Analyze historical delivery data and identify potential bottlenecks and inefficiencies.
*   Implement strategies to optimize logistics, such as negotiating better terms with providers or exploring alternative delivery options.
*   Gather more detailed data related to logistics and customer feedback to gain deeper insights into the causes of delivery delays.


## Folder Structure
*   **`data/`**: This directory contains the data used in the project, separated into three subdirectories:
    *   `raw/`: Contains the original, unaltered data files.
    *   `interim/`: Stores intermediate datasets created during data processing.
    *   `processed/`: Contains the final, cleaned, and processed datasets used for analysis.
*   **`src/`**: This directory houses the source code for the project, organized into modules:
    *   `analyzer/`: Contains modules for performing statistical analysis and generating insights.
    *   `data_processor/`: Includes modules for cleaning, transforming, and preparing the data.
    *   `interpreter/`: Contains modules for interpreting machine learning models and extracting meaningful information.
    *   `trainer/`: Includes modules for training machine learning models.
    *   `visualizer/`: Contains modules for creating visualizations and plots.
*   **`notebook/`**: This directory contains Jupyter notebooks used for data exploration, analysis, and prototyping:
    *   `data_analysis/`: Contains notebooks focused on in-depth data analysis and exploration.
    *   individual analysis notebooks: Contains other analysis notebooks

## Presentation Slides

*   [English Slides](https://drive.google.com/file/d/1dCPDZULL8augCY4Un5OoeSJYfKdU8kp1/view?usp=sharing)
*   [Japanese Slides](https://drive.google.com/file/d/1wuR52Ovcg05iJT5yFeyNzWCrwdA-1Hvp/view?usp=sharing)

## License

This project is licensed under the MIT License.
