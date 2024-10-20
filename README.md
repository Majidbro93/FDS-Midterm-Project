
# IMDb Top 1000 Movies Data Science Project

## Project Overview

This project is an analysis and visualization of the IMDb Top 1000 movies dataset, which contains various details about some of the most popular movies of all time. The analysis focuses on key factors like gross income, IMDb ratings, Metascores, movie certificates, and genres. By using data science techniques, this project provides insights into trends in the movie industry and aims to help production companies make better decisions based on audience preferences and revenue potential. Moreover, in terms of the technical data process point of view, this project used different imputatio techniques for finding the missing values in numerical and categorical variables in the dataset.

The project is divided into the following sections:
- **Data Cleaning and Imputation:** Handling missing values in the dataset, including Meta scores, gross income, and movie certificates.
- **Visualization:** Presenting the distribution of genres over time, the performance of top directors, and correlations between various movie features.
- **Interactive Streamlit Application:** Allowing users to explore the dataset interactively through visualizations and analysis.

The Streamlit app provides a user-friendly interface for exploring the IMDb Top 1000 dataset, allowing for filtering by genres, directors, and other key movie features.

## Features
- Analyze movie data based on IMDb ratings, Metascores, gross income, runtime, and more.
- Visualize trends in movie genres, gross income, and IMDb ratings over decades.
- Interactively explore the performance of top directors and actors.
- Imputation techniques for missing data using KNN and median methods.
- Data cleaning to standardize movie certificates and split movie genres into binary columns.
- Interactive visualizations of key features like gross income by year, movie certificates distribution, and correlation matrices.

## Setup Instructions

### Prerequisites
To run this project locally, you'll need the following:
- Python 3.8 or above
- `pip` package manager

### Required Libraries
Install the required libraries by running the following command:

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly wordcloud scikit-learn
```

### Dataset
You can download the IMDb Top 1000 dataset from [Kaggle](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows). The dataset used in this project has been cleaned and modified, and the final versions are provided in the following files:
- `imdb_1000_final.csv`
- `imdb_1000_genre.csv`
- `imdb_mapped_cert_Final.csv`

### How to Run the Project
1. Clone this repository or download the project files.
2. Navigate to the project directory.
3. Run the Streamlit app by executing the following command:

```bash
streamlit run Movies_project_streamlit_final.py
```

This will open the Streamlit application in your default web browser.

## Using the Streamlit Application

### Navigating the App
The app is divided into four sections accessible via the sidebar menu:
1. **Introduction:** Overview of the movie industry and project details.
2. **Data Process:** Explanation of data cleaning, handling missing values, and correlation analysis.
3. **Data Visualization:** Interactive plots to explore movie trends by genre, director, and movie certificates. 
4. **Conclusion:** Summary of findings and next steps.

### Interactive Features
- **Genre Selection:** Filter movies by genre to explore how specific genres have evolved over decades.
- **Director Performance:** Compare the performance of top directors based on gross income and IMDb ratings.
- **Certificate Analysis:** Visualize the distribution of movie certificates across different time periods.
- **Data Exploration:** Explore correlations between numerical features such as IMDb ratings, runtime, and gross income.

## Conclusion
This project demonstrates the power of data science in analyzing and understanding trends in the movie industry. It provides a platform for interactive exploration of the IMDb Top 1000 dataset, giving valuable insights for both movie enthusiasts and industry professionals.
