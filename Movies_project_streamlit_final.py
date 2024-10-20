
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px 

# Load the IMDB movies dataset
# We split the dataset in this project while we did analysis on the dataset

# Original datset hat was downloaded from the Kagle website
movies_df = pd.read_csv('imdb_top_1000.csv')
# Final dataset after all imutations
# I have corrected the standards fro the certificate after that and I have saved to other csv file
movies_final_def = pd.read_csv('imdb_1000_final.csv')
# dataset that with the separated genre column
movies_genre_def = pd.read_csv('imdb_1000_genre.csv')
# Corrected standrads for certfcate column based on the US 
movies_cert_mapped_def = pd.read_csv('imdb_mapped_cert_Final.csv')
# Corrected standards fro certificate column after mapping and dropping the ambiguous certificates
movies_cert_mapped_modified_def = pd.read_csv('imdb_mapped_cert_Final_modified.csv')
# Final dataset with the corrected certificate column
movies_cert_Final_with_correct_def = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')

# Data cleaning for runtime and gross columns
movies_df['Runtime'] = movies_df['Runtime'].str.replace(' min', '').astype(int)
movies_df['Gross'] = movies_df['Gross'].str.replace(',', '').astype(float)

# Sidebar menu for navigation
menu = st.sidebar.selectbox(
    "Select a section",
    ["Introduction","Data Process" ,"Data Visualization" , "Conclusion"]
)

# Introduction Section
if menu == "Introduction":
   # HTML for custom title
    import streamlit as st

# Define the clapper emoji
    clapper_emoji = "🎬"

    # HTML for custom title
    st.markdown("""
        <style>
        .custom-title {
            font-family: 'Pacifico', cursive;
            font-size: 50px;
            color: #ff6347;
        }
        </style>
        """, unsafe_allow_html=True)

    # Displaying the custom title with emoji
    st.markdown(f'<p class="custom-title">{clapper_emoji} Let\'s Talk About Movies!!</p>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    .custom-title {
        font-family: 'Pacifico', cursive;
        font-size: 30px;
        color: #ff6347;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p>By Majid Baradaran Akbarzadeh</p>', unsafe_allow_html=True)


    st.write("### Introduction")

    st.write("""
    In modern times, people have different tastes when it comes to choosing how they want to spend their free time. 
             Some prefer to spend their free time going out, reading books, or exercising. Others tend to watch series and movies a lot,
              not only because they can have a good time but also because they can learn many things from different cultures and science, or even enjoy simple comedies 
             without any special informative content, just for fun. Therefore, we can say that the film industry today plays a significant and influential role in people's lives, 
             shaping society’s culture, politics, and even economics. Furthermore, with the growing popularity of streaming platforms, the variety of genres, storytelling techniques, 
             and visual effects has made movies one of the most consumed forms of media worldwide.""")

    st.image("https://th.bing.com/th/id/R.5dd08addca8c9922408123292e2a5c3d?rik=0yiygbqJl0cAqQ&riu=http%3a%2f%2ftheseventhart.org%2fwp-content%2fuploads%2f2012%2f06%2fGodfatherIII.jpg&ehk=Em%2bD%2be0p08GrSa6RgxantpppC%2bynuuB2d7326mu3OMI%3d&risl=&pid=ImgRaw&r=0", caption="Al Pacino in a scene for GodFather III")     

    st.write("""As mentioned above, the movie industry has become incredibly important over the last few decades, and today it is a very serious business for many 
             companies and individuals around the globe. For example, you can see how the gross income of top-charting movies has increased throughout the years in the below figure. As a result, 
             many entertainment and movie corporations are investing in sophisticated plots, talented directors, and famous actors and actresses to gain more profit than their 
             competitors. Additionally, people's tastes in movie genres vary greatly, and audiences range in age from children to the elderly. In this regard, directors, producers, 
             and screenwriters must consider these factors when making movies in order to capture the audience's attention and maximize profit from their final product.""")
     

    # New Visualization 10: Total Gross Earnings by Year
    gross_by_year = movies_final_def.groupby('Released_Year')['Gross'].sum().reset_index()

    # Calculate the regression line
    slope, intercept = np.polyfit(gross_by_year['Released_Year'], np.log(gross_by_year['Gross']), 1)

    # Generate the regression line values
    regression_line = np.exp(slope * gross_by_year['Released_Year'] + intercept)

    # Create the Plotly figure with the Cividis color scale
    fig = px.line(gross_by_year, x='Released_Year', y='Gross', markers=True,
                title='Total Gross Earnings by Year',
                labels={'Released_Year': 'Release Year', 'Gross': 'Total Gross Earnings (in dollars)'},
                color_discrete_sequence=px.colors.sequential.Cividis)
    fig.update_traces(line=dict(width=2.5), marker=dict(size=8))
    fig.add_scatter(x=gross_by_year['Released_Year'], y=regression_line, mode='lines', name='Regression Line',
                    line=dict(color='red', width=2))

    # Set the layout and title font sizes
    fig.update_layout(title_font_size=16)
    fig.update_yaxes(type="log")
    fig.update_xaxes(title_font=dict(size=14, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=12, family='Arial', color='black', weight='bold'))
    fig.update_yaxes(title_font=dict(size=14, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=12, family='Arial', color='black', weight='bold'))
    fig.update_traces(hovertemplate='Year: %{x}<br>Gross: $%{y:,}<extra></extra>')
    st.plotly_chart(fig)



    st.write("""Given the importance of the movie industry, we chose to perform data analysis on a movie dataset. There are numerous datasets available on the internet 
             for such analysis, and we selected one of the most trusted and famous sources: IMDb's [Top 1000 Movies of All Time](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows), which provides most of the information we are looking for. 
             This dataset includes various types of data, such as movie certificates, overviews, IMDb ratings, Metacritic ratings, and gross income etc. By considering these 
             variables, we can conduct an informative analysis of what makes top-rated movies stand out and provide valuable insights for both audiences and movie production companies, 
             helping them decide where to invest their time and money. This analysis is useful for both ordinary moviegoers and specialized individuals (movie critics) who follow 
             the industry closely from artistic and technical perspectives.

Here, we present some initial results from the dataset to gain a better understanding of how many top-chart movies were directed by acclaimed directors 
             and which certificates are most common thoroughout the years. We have divided this project into four sections: Introduction, Data Processing, Visualizations, and Conclusion. 
             In the Data Processing section, we explain the statistical procedures we used to prepare our dataset for analysis, including Initial Data Analysis (IDA), 
             Exploratory Data Analysis (EDA), and data cleaning. 
             Additionally, in the Visualizations section, we present interactive results from the dataset, focusing on genres, director scores, and gross incomes based on genres and movie ratings. 
              Finally, in the Conclusion section, we highlight some key findings from our data analysis and the results we derived from 
             the dataset. LET'S BEGIN OUR JOURNEY INTO THE WORLD OF MOVIES! """)



    
    sns.set_style("whitegrid")
    colors = sns.color_palette("coolwarm", 10)
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))
    # Get the top 10 directors by number of movies
    top_directors = movies_final_def['Director'].value_counts().head(10)
    bars = ax.bar(top_directors.index, top_directors.values, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_title('Top Directors with Most Movies in Top 1000', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Director', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of Movies', fontsize=14, fontweight='bold', labelpad=10)

    # Rotate x-axis labels for better readability and increase font size
    ax.set_xticklabels(top_directors.index, rotation=45, ha='right', fontsize=12, fontweight='bold')

    # Add gridlines for better visualization, making them subtle and clean
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    # Remove top and right spines for a cleaner look
    sns.despine()

    # Add value labels on top of the bars for clarity
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, int(yval), ha='center', 
                fontsize=12, fontweight='bold', color='black')
    st.pyplot(fig)


    # Interactive Pie Chart (Movie Certificates)
    # Count the number of occurrences of each certificate
    certificate_counts = movies_cert_Final_with_correct_def['Certificate'].value_counts().reset_index()
    certificate_counts.columns = ['Certificate', 'Count']
    # Create the 3D-like pie chart
    fig = px.pie(certificate_counts, values='Count', names='Certificate',
                title='Distribution of Movie Certificates',
                hover_data=['Count'], hole=0.4)
    fig.update_traces(textinfo='percent+label', pull=[0.1] * len(certificate_counts), rotation=45)
    st.plotly_chart(fig)


###################################################################### Model Performance Section ############################################################################
elif menu == "Data Process":
        
        
    st.title("Data Process")

    st.write("""In this section, we will discuss how we conducted the Initial Data Analysis (IDA) and Exploratory Data Analysis (EDA) for our dataset in this project.
                Additionally, we will explain the steps we took for data cleaning and handling missing data in the dataset.""")

    # Display the first few rows of the dataset
    st.write("### Variables in the datset and type of the data")
    st.write("""We used the Top 1000 Movies of IMDb dataset for this project. In this regard, we introduce the variables we are working with in the project. 
                Below are the first few rows and columns of the dataset.""")
    st.write(movies_df.head())
    

    # Display the text with Streamlit
    st.markdown("""
        As you can see from the table above, we are working with **16 variables** in this dataset. These variables include:

        - **Poster_Link**: A link to the movie's poster
        - **Series_Title**: The movie title
        - **Released_Year**: The release year of the movie
        - **Certificate**: Indicates the authorized age range for viewing the movie
        - **Runtime**: The duration of the movie
        - **Genre**: The genres of the movie (some movies have multiple genres)
        - **IMDB_Rating**: The IMDb score of the movie
        - **Overview**: A brief summary of the movie
        - **Meta_score**: The Metacritic website score for each movie
        - **Star1 to Star4**: The actors and actresses who appeared in the movie
        - **No_of_Votes**: The number of people who voted for the movie
        - **Gross**: The gross revenue of the movie

        Now, we will introduce the types of data we are working with in this project, as shown in the table below.
        """)


    # Type of the data in the dataset
    df = pd.read_csv('imdb_1000_final.csv') 
    st.write("Data types of the columns in the dataset:")
    st.write(df.dtypes)
    st.write("""As you can see, out of the 16 different variables we are working with in this project, only six are numerical, and two of these numerical variables—released year and runtime—are categorical in nature. 
                Additionally, both certificate and genres are also categorical variables.""")
        
    st.write("### Heatmap of Missing Values")
    
    st.write("""Next, the heatmap for the missing values in the original dataset (before imputation and cleaning) is presented below. From the heatmap, 
                we can see that the missing values are in the Certificate (categorical data), Meta_score (numerical data), and Gross (numerical data) columns. 
                In the remainder of this section, we will explain how we handled the missing values and the 
                different procedures we followed to prepare the dataset for imputation using various methods.""")    

        # Visualization 1: Heatmap of Missing Values
    st.write("""Below figure shows the missing values in the original dataset (top 1000 IMDB movies). Based on the heatmap figure for missing values, we see that we have the missing 
        data.""")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(movies_df.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title('Missing Values Heatmap For Original Dataset')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("### Correlation Matrix For Original Dataset")
    st.write("""The figure below shows the correlation matrix for the numerical variables in the dataset we are working on. 
                The purpose of this is to gain an initial understanding of how the numerical variables are related to each other. 
                After imputation, we aim to preserve these relationships as much as possible, 
                ensuring that the correlations between the variables are not drastically altered.""")
        
        # Visualization 2: Correlation Matrix
    numerical_columns = ['IMDB_Rating', 'Runtime', 'Gross', 'Meta_score', 'No_of_Votes']
    corr_matrix = movies_df[numerical_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

    st.write("### Data Cleaning ")
    st.write("""In our dataset, we do not have any duplicate numerical data. However, in the Certificate column, which indicates the movie's viewing allowance for specific audiences (age range), we have ratings listed with different standards, each with its own sign and convention. Another issue arises in the Genre column, where some movies are assigned multiple genres, separated by commas. This creates a challenge for the imputation process, as we need to separate these genres into different columns. To address
                both issues, we first used mapping to standardize the sign conventions in the Certificate column and then split the Genre column 
                into separate columns for each genre.""")

        # Creating a count plot to visualize the relationship between 'Genre_1' and 'Certificate'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=movies_genre_def, x='Genre_1', hue='Certificate')
    plt.title("Distribution of 'Main Genre' across different 'Certificate' categories before mapping")
    plt.xlabel("Primary Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    st.write("""As you see from the figure above, we have different sign and convention for movies ratings based on different age ranges. In this regard, we considered the movie ratings 
        that used in the United States which we can see them on the IMDB website in the sction of movie ratings. """)
    st.markdown("""
        These ratings are as follows:

        - **G**: For all audience
        - **PG**: Parental Guidance Suggested (mainly for under 10's)
        - **PG-13**: Parental Guidance Suggested for children under 13
        - **R**: Under 17 not admitted without parent or guardian
        - **Approved**: Pre-1968 titles only (from the MPA site) Under the Hays Code, films were simply approved or disapproved based on whether they were deemed 'moral' or 'immoral'.)
        """)
    st.write("""So, we mapped all the other ratings in the dataset based on the same or similar categories used in the United States. Moreover, some ratings, 
                such as Passed, UA, and U/A, were considered ambiguous in meaning and definition. Therefore, we treated them as NaN and will attempt to impute them using 
                the classifier imputer technique called Random Forest. Below, you can see the bar plot showing the count of different certificates
                based on the main genre of the film after mapping.""")
        
        # Creating a count plot to visualize the relationship between 'Genre_1' and 'Certificate'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=movies_cert_mapped_def, x='Genre_1', hue='Certificate')
    plt.title("Distribution of 'Genre_1' across different 'Certificate' categories after mapping")
    plt.xlabel("Primary Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    # Splitting the 'Genre' column into multiple columns based on ','
    genre_split = movies_df['Genre'].str.split(',', expand=True)
    genre_split.columns = [f'Genre_{i+1}' for i in range(genre_split.shape[1])]
    imdb_1000_gensp_df = pd.concat([movies_df, genre_split], axis=1)
        # Plot heatmap to visualize missing values
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(movies_cert_mapped_modified_def.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Heatmap of Missing Values of the gensep dataset')
    plt.xticks(rotation=45)
    st.pyplot(fig)

        # Load your IMDb dataset
    df = movies_cert_mapped_modified_def

        # Count missing values in each column
    missing_values = df.isnull().sum()

        # Display the missing values count in Streamlit
    st.write("### Missing Values Count")
    st.write(missing_values)



    st.write("### Handling missing values")
    st.write("""We have missing values in three columns, as shown in the heatmap for missing data. Specifically, there are missing values 
                in the Certificate, Meta_score, and Gross columns of the dataset. To address this issue, we have considered several methods for handling the missing values in these columns.
                First, we will explain how we perform imputation for the numerical columns, such as Meta_score and Gross.""")
    st.write("### Handling missingness in Meta score column")
    st.write("""In this section we will discuss that how we impute the missing values in the Meta_score column by using two diffrent methods. These two approaches are
                Median and K-nearest neighbor (KNN). """)
    st.write("""In the plots below, we see two different comparisons of the dataset before and after imputation using the Median and KNN methods. Our KNN imputation was based on the 
                IMDB_Rating, No_of_Votes, and Meta_score columns. Since IMDB_Rating and Meta_score are closely related for the top 1000 movies on the IMDb chart, we applied the median method 
                based on the IMDB_Rating column. From the histograms, we observe a better spread with the KNN method compared to the median method, where the peak values are higher. Additionally,
                in the scatter plots, 
                we again see a better spread with the KNN method compared to the median method, where the latter shows straight lines of data points among the original dataset's data points.
                """)
        
    ### Using KNN for findinf the missingness in the Meta_score column
    from sklearn.impute import KNNImputer
        

    # Selecting relevant numerical features
    features = ['IMDB_Rating', 'No_of_Votes', 'Meta_score']  # Features including 'Meta_score'
    df_knn = imdb_1000_gensp_df[features].copy()

        # Applying KNN Imputation
    imputer = KNNImputer(n_neighbors=10)
    df_knn_imputed = pd.DataFrame(imputer.fit_transform(df_knn), columns=features)


    imdb_1000_KNNMeta_df = imdb_1000_gensp_df.copy()

    imdb_1000_KNNMeta_df['Meta_score'] = df_knn_imputed['Meta_score']
        
        ## Using Median for finding missingness in Meta_score column


    imdb_1000_MedMeta_df = imdb_1000_gensp_df.copy()
        # Fill missing Meta_score values with the median Meta_score within each genre in the copied dataset
    imdb_1000_MedMeta_df['Meta_score'] = imdb_1000_MedMeta_df.groupby('IMDB_Rating')['Meta_score'].transform(lambda x: x.fillna(x.median()))

        ## Plots for comparing KNN and median imoutation for Meta_score column

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot for the KNN imputed Meta_score on the first subplot
    imdb_1000_gensp_df['Meta_score'].plot(kind='hist', alpha=0.5, color='blue', label='Original (Missing Values)', bins=30, ax=axes[0])
    imdb_1000_KNNMeta_df['Meta_score'].plot(kind='hist', alpha=0.5, color='green', label='Imputed (KNN)', bins=30, ax=axes[0])
    axes[0].set_title('Histogram Comparison of Meta_score (KNN Imputation)')
    axes[0].set_xlabel('Meta_score')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

        # Histogram Plot for the Median imputed Meta_score on the second subplot
    imdb_1000_gensp_df['Meta_score'].plot(kind='hist', alpha=0.5, color='blue', label='Original (Missing Values)', bins=30, ax=axes[1])
    imdb_1000_MedMeta_df['Meta_score'].plot(kind='hist', alpha=0.5, color='green', label='Imputed (Median)', bins=30, ax=axes[1])
    axes[1].set_title('Histogram Comparison of Meta_score (Median Imputation)')
    axes[1].set_xlabel('Meta_score')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()


    plt.tight_layout()
    st.pyplot(fig)

        # Scattered plots for comparing the two methods for the imoutation on Meta_core
        

    imputed_mask = imdb_1000_gensp_df['Meta_score'].isnull()


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot for KNN imputed Meta_score on the first subplot
    axes[0].scatter(imdb_1000_gensp_df.index[~imputed_mask], imdb_1000_KNNMeta_df.loc[~imputed_mask, 'Meta_score'], 
                        color='blue', label='Original Values', alpha=0.6)
    axes[0].scatter(imdb_1000_gensp_df.index[imputed_mask], imdb_1000_KNNMeta_df.loc[imputed_mask, 'Meta_score'], 
                        color='red', label='Imputed Values by KNN', alpha=0.6)
    axes[0].set_title('Scatter Plot: Original vs Imputed Meta_score (KNN)')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Meta_score')
    axes[0].legend()

        # Scatter plot for Median imputed Meta_score on the second subplot
    axes[1].scatter(imdb_1000_gensp_df.index[~imputed_mask], imdb_1000_MedMeta_df.loc[~imputed_mask, 'Meta_score'], 
                        color='blue', label='Original Values', alpha=0.6)
    axes[1].scatter(imdb_1000_gensp_df.index[imputed_mask], imdb_1000_MedMeta_df.loc[imputed_mask, 'Meta_score'], 
                        color='red', label='Imputed Values by Median', alpha=0.6)
    axes[1].set_title('Scatter Plot: Original vs Imputed Meta_score (Median)')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Meta_score')
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)



    st.write("""In the scatter plots below, we compare the imputed Meta_score data points using the KNN and median methods. We observe a 
                better trend and distribution of the imputed data points with the KNN method compared to the median method. Although the error bounds for 
                the KNN and median methods are similar, at 15.35% and 
                13.93%, respectively, the distribution trend of the KNN method is better than that of the median method.""")


    # Scatter plot median Vs KNN for Meta_score in terms of No_of_votes

        # Create a mask to identify the imputed values (previously missing values)
    imputed_mask = imdb_1000_gensp_df['Meta_score'].isnull()

        # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot for KNN imputed Meta_score on the first subplot
    axes[0].scatter(imdb_1000_KNNMeta_df.loc[~imputed_mask, 'No_of_Votes'], imdb_1000_KNNMeta_df.loc[~imputed_mask, 'Meta_score'], 
                        color='blue', label='Original Values', alpha=0.6)
    axes[0].scatter(imdb_1000_KNNMeta_df.loc[imputed_mask, 'No_of_Votes'], imdb_1000_KNNMeta_df.loc[imputed_mask, 'Meta_score'], 
                        color='red', label='Imputed Values by KNN', alpha=0.6)
    axes[0].set_title('Scatter Plot: Original vs Imputed Meta_score (KNN)')
    axes[0].set_xlabel('No of Votes')
    axes[0].set_ylabel('Meta_score')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True)

        # Scatter plot for Median imputed Meta_score on the second subplot
    axes[1].scatter(imdb_1000_MedMeta_df.loc[~imputed_mask, 'No_of_Votes'], imdb_1000_MedMeta_df.loc[~imputed_mask, 'Meta_score'], 
                        color='blue', label='Original Values', alpha=0.6)
    axes[1].scatter(imdb_1000_MedMeta_df.loc[imputed_mask, 'No_of_Votes'], imdb_1000_MedMeta_df.loc[imputed_mask, 'Meta_score'], 
                        color='red', label='Imputed Values by Median', alpha=0.6)
    axes[1].set_title('Scatter Plot: Original vs Imputed Meta_score (Median)')
    axes[1].set_xlabel('No of Votes')
    axes[1].set_ylabel('Meta_score')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True)

        
    plt.tight_layout()

    st.pyplot(fig)


    ###### Gross imputation
    st.write("### Handling missingness in Gross column")
    st.write("""In this section, we applied the same procedure to the Gross income column using two different methods: the Median and K-nearest neighbor (KNN) approaches. """)
    st.write("""Based on the plots below, we observe a better trend for imputing the missing data using KNN for the Gross income of the movies, 
                whereas with the median method, only one point was imputed. Additionally, from the scatter plot using KNN for imputation, we see that the 
                imputed data points follow the same trend as the general dataset.
                Therefore, for Gross income, we chose KNN as the better imputation method.""")
    ## KNN method
        # Selecting relevant numerical features
    features = ['IMDB_Rating', 'No_of_Votes', 'Gross']  # Features including 'Meta_score'
    df_knn = imdb_1000_gensp_df[features].copy()

        # Applying KNN Imputation
    imputer = KNNImputer(n_neighbors=10)
    df_knn_imputed = pd.DataFrame(imputer.fit_transform(df_knn), columns=features)


    imdb_1000_KNNGross_df = imdb_1000_gensp_df.copy()

    imdb_1000_KNNGross_df['Gross'] = df_knn_imputed['Gross']

        ## Median method

    imdb_1000_MedGross_df = imdb_1000_gensp_df.copy()
        # Fill missing Meta_score values with the median Meta_score within each genre in the copied dataset
    imdb_1000_MedGross_df['Gross'] = imdb_1000_MedGross_df.groupby('No_of_Votes')['Gross'].transform(lambda x: x.fillna(x.median()))


        # Create a mask to identify the imputed values (previously missing values)
    imputed_mask = imdb_1000_gensp_df['Gross'].isnull()

        # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot for KNN imputed Gross on the first subplot
    axes[0].scatter(imdb_1000_gensp_df.index[~imputed_mask], imdb_1000_KNNGross_df.loc[~imputed_mask, 'Gross'], 
                        color='blue', label='Original Values', alpha=0.6)
    axes[0].scatter(imdb_1000_gensp_df.index[imputed_mask], imdb_1000_KNNGross_df.loc[imputed_mask, 'Gross'], 
                        color='red', label='Imputed Values by KNN', alpha=0.6)
    axes[0].set_title('Scatter Plot: Original vs Imputed Gross (KNN)')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Gross')
    axes[0].set_yscale('log')
    axes[0].legend()

    # Scatter plot for Median imputed Gross on the second subplot
    axes[1].scatter(imdb_1000_gensp_df.index[~imputed_mask], imdb_1000_MedGross_df.loc[~imputed_mask, 'Gross'], 
                    color='blue', label='Original Values', alpha=0.6)
    axes[1].scatter(imdb_1000_gensp_df.index[imputed_mask], imdb_1000_MedGross_df.loc[imputed_mask, 'Gross'], 
                    color='red', label='Imputed Values by Median', alpha=0.6)
    axes[1].set_title('Scatter Plot: Original vs Imputed Gross (Median)')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Gross')
    axes[1].set_yscale('log')
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)


    imputed_mask = imdb_1000_gensp_df['Gross'].isnull()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot for KNN imputed Gross on the first subplot
    axes[0].scatter(imdb_1000_KNNGross_df.loc[~imputed_mask, 'No_of_Votes'], imdb_1000_KNNGross_df.loc[~imputed_mask, 'Gross'], 
                    color='blue', label='Original Values', alpha=0.6)
    axes[0].scatter(imdb_1000_KNNGross_df.loc[imputed_mask, 'No_of_Votes'], imdb_1000_KNNGross_df.loc[imputed_mask, 'Gross'], 
                    color='red', label='Imputed Values by KNN', alpha=0.6)
    axes[0].set_title('Scatter Plot: Original vs Imputed Gross (KNN)')
    axes[0].set_xlabel('No of Votes')
    axes[0].set_ylabel('Gross')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True)

    # Scatter plot for Median imputed Gross on the second subplot
    axes[1].scatter(imdb_1000_MedGross_df.loc[~imputed_mask, 'No_of_Votes'], imdb_1000_MedGross_df.loc[~imputed_mask, 'Gross'], 
                    color='blue', label='Original Values', alpha=0.6)
    axes[1].scatter(imdb_1000_MedGross_df.loc[imputed_mask, 'No_of_Votes'], imdb_1000_MedGross_df.loc[imputed_mask, 'Gross'], 
                    color='red', label='Imputed Values by Median', alpha=0.6)
    axes[1].set_title('Scatter Plot: Original vs Imputed Gross (Median)')
    axes[1].set_xlabel('No of Votes')
    axes[1].set_ylabel('Gross')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

###### Certificate imputation
    st.write("### Handling missingness in Certificate column")
    st.write("""In this section, we will explain the process we followed to identify the missing values in the Certificate column. Since the Certificate column is not numeric, 
            we needed to use a method that handles missing values for categorical data. Here, we used the Random Forest Classifier (RFC) to impute the missing values in this column. 
            For this, we encoded variables such as Genre_1 (main genre of the movie), Director, Star1 (leading actor), Star2 (supporting actor 1), Star3 (supporting actor 2), and Star4 
            (supporting actor 3). Using the RFC method,
            we divided the dataset into training and test data points and allowed the model to learn how to predict the correct Certificate based on the mentioned variables.""")


    ## Comparing the correlation matrices

    

    # Ensure 'Gross' and 'Runtime' are strings before replacing characters for the original dataset
    movies_df['Gross'] = movies_df['Gross'].astype(str).str.replace(',', '').astype(float)
    movies_df['Runtime'] = movies_df['Runtime'].astype(str).str.replace(' min', '').astype(float)

    # Ensure 'Gross' and 'Runtime' are strings before replacing characters for the imputed dataset
    movies_final_def['Gross'] = movies_final_def['Gross'].astype(str).str.replace(',', '').astype(float)
    movies_final_def['Runtime'] = movies_final_def['Runtime'].astype(str).str.replace(' min', '').astype(float)

    # Select relevant columns for correlation
    selected_columns = ['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime', 'Gross']

    correlation_matrix_org = movies_df[selected_columns].corr()

    correlation_matrix_gensp = movies_final_def[selected_columns].corr()

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the correlation matrix for original dataset
    sns.heatmap(correlation_matrix_org, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Correlation Matrix: Original Dataset")

    # Plot the correlation matrix for imputed dataset
    sns.heatmap(correlation_matrix_gensp, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Correlation Matrix: Imputed Dataset")

    plt.tight_layout()
    st.pyplot(fig)




    # Plot heatmap for the final dataset
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(movies_final_def.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Heatmap of Missing Values of the gensep dataset')
    plt.xticks(rotation=45)
    st.pyplot(fig)










###########################################################################################################################################################
############################################################## Data Visualization Section #################################################################
###########################################################################################################################################################
elif menu == "Data Visualization":
    
    st.title("Vilsualizations of the IMDB Top 1000 Movies")

    st.write("""In this section, we will provide you with some informative and interesting results through interactive visualizations. Our analysis was based on the variables in the top 1000 movies dataset from IMDb. In this regard, our focus was on how different genres of movies evolved and how they can affect the movie market in terms of gross revenue and scores from IMDb and Metacritic. Moreover, we have analyzed the performance of different directors and actors, examining how they influence movie performance based on the same characteristics, such as movie scores and income. We should mention that we designed this application to cater to a wide audience, ranging from ordinary people who want to gain some knowledge about the movie industry to those in charge of movie corporations,
              helping them decide which movie genre, certificate, director, and actor are worth investing in to gain the most profit and make their final
              product stand out from the crowd.""")


    st.write("### Top Genres Over the Decades and their popularity")
    
    st.write("""The graph below shows a histogram plot for different genres to visualize the rate of change in the number of movies produced with a specific genre over different decades. You can select the genre you want from the drop-down menu and see how many movies of that specific genre entered the market and charted in the top 1000. For example, with the advent of new technology in the movie industry, we observe an increase in the number of Sci-Fi movies produced from 1980 to 2020. This indicates that the movie industry decided to produce more Sci-Fi movies as they became more feasible. Additionally, 
             you can compare the gross revenue of each genre over the decades simultaneously. This allows you to observe the correlation between movie genres and their
              gross income over the years.  """)


    # Load dataset
    df = pd.read_csv('imdb_1000_final.csv')  # Ensure your file path is correct
    # Split the 'Genre' column
    df_genres = df.assign(Genre=df['Genre'].str.split(', ')).explode('Genre')
    # Convert 'Released_Year' and 'Gross' to numeric
    df_genres['Released_Year'] = pd.to_numeric(df_genres['Released_Year'], errors='coerce')
    df_genres['Gross'] = pd.to_numeric(df_genres['Gross'], errors='coerce')
    # Create a new column to group movies into decades
    df_genres['Decade'] = (df_genres['Released_Year'] // 10) * 10
    available_genres = df_genres['Genre'].unique().tolist()
    st.write("Select genres to compare:")
    genres_to_compare = st.multiselect("Choose genres", available_genres, default=[available_genres[0],available_genres[6]])

    # Visualization 1: Genres over decades (Histogram of released years)
    if genres_to_compare:
        df_selected_genres = df_genres[df_genres['Genre'].isin(genres_to_compare)]
        decade_bins = np.arange(1920, 2030, 10)  
        plt.figure(figsize=(12, 7))

        sns.histplot(data=df_selected_genres, x='Released_Year', hue='Genre', hue_order=genres_to_compare,
                    bins=decade_bins, multiple='dodge', shrink=0.8, palette='Set2', edgecolor='black', linewidth=1.2)

        plt.xlabel('Released Year (by Decade)', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Movies', fontsize=14, fontweight='bold')
        plt.title('Comparison of Released Years (by Decade) for Selected Genres', fontsize=18, fontweight='bold', pad=20)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(title='Genre', fontsize=12, title_fontsize=14)
        st.pyplot(plt.gcf())

        # Visualization 2: Genres popularity ("Decades vs. Gross Income for Selected Genres")
        plt.figure(figsize=(12, 7))
        gross_by_decade = df_selected_genres.groupby(['Decade', 'Genre'])['Gross'].sum().reset_index()
        sns.lineplot(data=gross_by_decade, x='Decade', y='Gross', hue='Genre', marker='o', palette='Set2')
        plt.xlabel('Decade', fontsize=14, fontweight='bold')
        plt.ylabel('Total Gross Income', fontsize=14, fontweight='bold')
        plt.title('Decades vs. Gross Income for Selected Genres', fontsize=18, fontweight='bold', pad=20)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(title='Genre', fontsize=12, title_fontsize=14)
        st.pyplot(plt.gcf())
    else:
        st.write("Please select at least one genre from the dropdown above.")


#################################################
#################################################
##################################################

    st.write("### Gross Income and Number of Votes by Certificate Over Decades")
    st.write("""The graph below shows the kernel density estimate (KDE) plot, illustrating how the data is distributed for different movie certificates 
             in terms of either gross revenue or the number of votes by people on IMDb. The peak values for each movie certificate indicate that a specific certificate gained popularity 
             during a particular decade based on the above-mentioned variables. Moreover, you may observe shifts in specific certificates over the range of decades, revealing how the 
             popularity of certain movie certificates increased or decreased over time. Additionally, the width of the plots indicates
              how many movies with a specific certificate were produced. Furthermore, you can change the y-axis to switch between gross revenue and the number of votes from the audience.""")



# Visualization 3: Normalized Gross Revenue and Normalized Number of Votes by Certificate Over Decades

    # Load your IMDb dataset
    df = pd.read_csv('imdb_1000_final_with_correct_certificate.csv')

    # Convert 'Released_Year', 'Gross', and 'No_of_Votes' to numeric
    df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
    df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')
    df['No_of_Votes'] = pd.to_numeric(df['No_of_Votes'], errors='coerce')

    # Create a new column to group movies into decades
    df['Decade'] = (df['Released_Year'] // 10) * 10

    # Normalize 'Gross' and 'No_of_Votes'
    df['Normalized Gross Revenue'] = df['Gross'] / df['Gross'].sum()
    df['Normalized Number of Votes'] = df['No_of_Votes'] / df['No_of_Votes'].sum()

    available_certificates = df['Certificate'].unique().tolist()

    # Allow certificate selection
    st.write("Select certificates to compare:")
    certificates_to_compare = st.multiselect("Choose certificates", available_certificates, 
                                            default=[available_certificates[0], available_certificates[2], available_certificates[4]], key='ghgh11111')
    y_axis_option = st.radio("Select the metric to plot on the Y-axis:", ('Normalized Gross Revenue', 'Normalized Number of Votes'))
    
    if certificates_to_compare:
        df_selected_certificates = df[df['Certificate'].isin(certificates_to_compare)]


        palette = sns.color_palette("husl", len(certificates_to_compare))

        # Plot the KDE curve with shading
        plt.figure(figsize=(10, 6))

        for idx, certificate in enumerate(certificates_to_compare):
            df_certificate = df_selected_certificates[df_selected_certificates['Certificate'] == certificate]

            total_weight = df_certificate[y_axis_option].sum()
            normalized_weights = df_certificate[y_axis_option] / total_weight if total_weight > 0 else df_certificate[y_axis_option]

            # Plot the KDE curve with normalized weights
            sns.kdeplot(data=df_certificate, x='Decade', weights=normalized_weights, fill=True, 
                        color=palette[idx], label=f'{certificate}', alpha=0.6)
        plt.xlabel('Decade', fontweight='bold')  
        plt.ylabel(y_axis_option, fontweight='bold')  
        plt.title(f'Distribution of {y_axis_option} over Decades for Selected Certificates', fontweight='bold', fontsize=14)  # Bold title
        plt.legend(title='Certificate', title_fontsize='13', fontsize='11', loc='upper right')  # Bold legend title and larger font

        plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3) 
        st.pyplot(plt.gcf())
    else:
        st.write("Please select at least one certificate from the menu above.")



###############################################################################################
# Visualization 4: comparing directors IMDB scores and gross income for different movies

    st.write("### Top directors performance over time based on gross revenue and IMDB ratings")
    st.write("""This is one of the most interesting graphs in the application. We can see how the performance of top directors changed over time using a dual axis, 
             displaying both the gross revenue of their movies and IMDb scores. If you hover over each point on the graph, you can see additional information, such as 
             the name of the movie directed by that specific director. Here, we observe that movies with higher IMDb scores tend to generate less revenue compared to the 
             director's other movies or even compared to other directors' films. In our view, this may be due to several reasons, such as movies with higher IMDb scores gaining 
             attention after their theatrical release or becoming popular years later from the audience's perspective. Conversely, we notice the opposite trend for movies with lower 
             scores but higher gross revenue. Moreover, some directors performed better with certain movies, while others failed to capture the audience's attention 
             with their other films, which explains the trends seen in this visualization.""")

    import plotly.graph_objects as go

    df = pd.read_csv('imdb_1000_final.csv')

    df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
    df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')

    # Sidebar for director selection
    directors = df['Director'].unique().tolist()
    st.write("Select Directors for Comparing IMDb Ratings and Gross Income")
    selected_directors = st.multiselect("Select directors", directors, default=[directors[1],directors[6]])

    # Filter the dataset based on the selected directors
    df_directors = df[df['Director'].isin(selected_directors)]

    # Sort the movies by release year
    df_directors = df_directors.sort_values(by='Released_Year')

    # Plot the IMDb rating and gross income trend over time for the selected directors
    if not df_directors.empty:
        # Create the figure using Plotly Graph Objects
        fig = go.Figure()

        for director in selected_directors:
            df_director = df_directors[df_directors['Director'] == director]

            fig.add_trace(
                go.Scatter(x=df_director['Released_Year'], y=df_director['IMDB_Rating'], 
                        mode='lines+markers', name=f'IMDb Rating: {director}', line=dict(shape='linear'),
                        hovertemplate='<b>%{text}</b><br>IMDb Rating: %{y}<br>Year: %{x}',
                        text=df_director['Series_Title'])
            )

            fig.add_trace(
                go.Scatter(x=df_director['Released_Year'], y=df_director['Gross'], 
                        mode='lines+markers', name=f'Gross Income: {director}', line=dict(shape='linear', dash='dash'),
                        yaxis='y2',  # This specifies the second y-axis
                        hovertemplate='<b>%{text}</b><br>Gross: $%{y}<br>Year: %{x}',
                        text=df_director['Series_Title'])
            )

        fig.update_layout(
            title="IMDb Ratings and Gross Income over Time for Selected Directors",
            xaxis_title="Released Year",
            yaxis_title="IMDb Rating",
            yaxis2=dict(title="Gross Income ($)", overlaying='y', side='right'),
            legend_title="Metrics",
            hovermode="x unified"
        )
        st.plotly_chart(fig)

    else:
        st.write("Please select at least one director.")

    

    
    
    
# Visualization 5: 


    st.write("### Top IMDb Rating Movies by Star")
    st.write("""For this part of the visualization, we aim to provide an overview of each actor or actress in this dataset. By selecting your favorite movie star, 
             you can view their movie title(s), gross revenue, release date, IMDb rating, and the movie’s rank in the top IMDb chart. This gives a brief resume of that specific 
             movie star and the quality of the movies they have appeared in. Moreover, since this application is open to the public, producers and movie corporations can use 
             this part of the application for a quick review of the person they are considering for a contract. In this regard, it provides convenient access to the background of the movie star.""")


###########################################
    

    # Star selection
    star = st.selectbox('🌟 Select your favorite celebrity', pd.concat([movies_final_def['Star1'], movies_final_def['Star2'], movies_final_def['Star3'], movies_final_def['Star4']]).unique(), key="u2")

    # Filter the dataset for movies where the actor appeared
    filtered_df = movies_final_def[
        (movies_final_def['Star1'] == star) | 
        (movies_final_def['Star2'] == star) | 
        (movies_final_def['Star3'] == star) | 
        (movies_final_def['Star4'] == star)
    ].sort_values('IMDB_Rating', ascending=False)

    # Display top movies by IMDb rating
    st.markdown(f"## **Top Movies for _{star}_ based on IMDb Rating**")

    styled_df = filtered_df[['Series_Title', 'IMDB_Rating', 'Released_Year', 'Gross']].style \
        .format({
            'IMDB_Rating': '{:.1f}', 
            'Gross': '${:,.0f}'  
        }) \
        .set_properties(**{
            'background-color': '#F5F5F5',  
            'color': 'black',  
            'border-color': '#7F7F7F',  
            'font-size': '15px',  
            'font-family': 'Verdana',  
            'border': '2px solid #7F7F7F'  
        }) \
        .highlight_max(subset='IMDB_Rating', color='lightgreen') \
        .highlight_min(subset='IMDB_Rating', color='lightcoral') \
        .bar(subset='Gross', color='skyblue', vmin=0) \
        .set_table_styles([{
            'selector': 'thead th',  
            'props': [('background-color', '#006400'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]
        }, {
            'selector': 'tbody tr:nth-child(even)', 
            'props': [('background-color', '#EFEFEF')]
        }, {
            'selector': 'tbody tr:nth-child(odd)',  
            'props': [('background-color', 'white')]
        }, {
            'selector': 'th.col_heading',  
            'props': [('background-color', '#D3D3D3'), ('color', 'black'), ('font-weight', 'bold'), ('font-style', 'italic')]
        }, {
            'selector': 'tbody td',  
            'props': [('padding', '10px')]  
        }])

    
    st.write(styled_df)




# Visualization 6: 
    st.write("### Correlation between some numerical attributes of the dataset")
    st.write("""In this section, we aim to analyze some numerical variables in the dataset. In the first plot, you can choose between four different variables in the dataset, 
             which will give you some insights into how each one correlates with the others. For example, if we add all of them at the same time, we see that movies with longer 
             durations tend to have lower gross revenue. Conversely, they tend to receive higher (better) IMDb scores. However, in terms of their Metascore, we observe that although 
             many movies with higher IMDb scores also receive higher Metascores, this is not always the case. Furthermore, a similar trend is visible in the second interactive plot, 
             which shows a scatter plot between IMDb scores and Metacritic scores. As a reminder, IMDb scores are derived from ordinary people, who may not necessarily have extensive
              knowledge of movies from a technical standpoint. In contrast, Metacritic scores are typically provided by professional
              film critics, whose opinions tend to be stricter than IMDb ratings. As a result, we observe some movies with good IMDb scores that did not receive high Metascores.""")



    ########################################
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    # Load the IMDb dataset
    df = pd.read_csv('imdb_1000_final.csv')

    # Sidebar Inputs for attribute selection
    attributes = st.multiselect(
        ' **Select Attributes** to Visualize', 
        ['IMDB_Rating', 'Gross', 'Runtime', 'Meta_score'], 
        default=['IMDB_Rating', 'Gross', 'Runtime']  # Some default attributes selected
    )

    # Drop rows with missing values in selected attributes
    df_selected = df.dropna(subset=attributes)

    if attributes:
        # Parallel Coordinates Plot
        fig = px.parallel_coordinates(
            df_selected,
            color="IMDB_Rating",
            dimensions=attributes,
            labels={col: col.replace('_', ' ') for col in attributes},  # Cleaner axis labels
            color_continuous_scale=px.colors.sequential.Viridis,  # Fancy color scale
            range_color=[df_selected['IMDB_Rating'].min(), df_selected['IMDB_Rating'].max()],
        )
        
        fig.update_layout(
            title={
                'text': '',
                'y': 0.9,  
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=14, family="Verdana", color="black", weight='bold')  
            },
            font=dict(family="Verdana", size=12, color="black"),  
            coloraxis_colorbar=dict(
                title="IMDb Rating",
                titlefont=dict(size=12, weight='bold'),  
                thicknessmode="pixels",
                thickness=20,
                lenmode="fraction",
                len=0.75,
                yanchor="middle",
                y=0.5
            ),
            plot_bgcolor='white',  
            paper_bgcolor='white',  
        )


        fig.update_traces(line=dict(colorbar=dict(thickness=15)))

        st.plotly_chart(fig)

    else:
        st.write("Please select at least one attribute to visualize.")



    ###########################################



    # New Visualization 7: Interactive Scatter Plot (Meta Score vs. IMDB Rating)
    fig = px.scatter(
        movies_df, 
        x='Meta_score', 
        y='IMDB_Rating',
        hover_data=['Series_Title', 'Released_Year'],  
        title=" IMDB Rating vs. Meta Score",  
        labels={'Meta_score': 'Meta Score', 'IMDB_Rating': 'IMDB Rating'},  
        color='IMDB_Rating',  
        color_continuous_scale=px.colors.sequential.Teal  
    )


    fig.update_layout(
        title={
            'text': "🎬 IMDB Rating vs. Meta Score",
            'y': 0.9,  
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, family="Verdana", color="black", weight='bold')  
        },
        font=dict(family="Verdana", size=12, color="black"),  
        xaxis_title=dict(text='Meta Score', font=dict(size=14, weight='bold')),  
        yaxis_title=dict(text='IMDB Rating', font=dict(size=14, weight='bold')),  
        
        
        xaxis=dict(
            tickfont=dict(size=12, weight='bold'),  
        ),
        yaxis=dict(
            tickfont=dict(size=12, weight='bold'),  
        ),
        
        plot_bgcolor='white',  
        paper_bgcolor='white', 
        hoverlabel=dict(font_size=12, font_family="Arial", font_color="black")  
    )

    
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    st.markdown("**💡 Tip:** _Hover over the plane on each point to see its details better!_")













elif menu == "Conclusion":
    st.title("Conclusions")

    st.write("""In this section, we will discuss the conclusions and key findings from this project. So far, we have provided a detailed, 
             step-by-step explanation of the processes we followed, including discussing the movie industry, introducing our dataset, 
             performing data processing to prepare it for analysis, and presenting our results through interactive plots. 
             These plots give us deeper insights into what is happening in our dataset and, to a broader extent, 
             offer some initial ideas about trends in this fascinating industry.""")
    
    st.write("""Moreover, our initial goal in this project was to build an application for movie production corporations to use these analyses to make better decisions 
             for capturing a larger share of the movie market. In this regard, we presented our results on the correlations between variables, such as how movie genres, 
             certificates, directors, and actors can play an important role in increasing income and receiving high scores from both general audiences and professional critics.""")
    
    st.write("""As a result, from a data scientist's point of view, we have found that some emerging genres in the movie industry can significantly increase a movie's gross revenue and capture people's 
             attention over time. This is due to the fact that technological developments in the movie industry have allowed corporations to produce movies in new genres, such as 
             Sci-Fi and adventure, which are novel to audiences. Additionally, we observed how different movie certificates have evolved over time. This provides valuable information 
             for investors in the industry, helping them choose which types of plots are trending at the moment. Furthermore, we analyzed the backgrounds of directors and actors who 
             appeared in top-charted movies and examined the trends between their movies' gross income and scores. Lastly,
              we conducted an analysis of correlations between various parameters in the movies, revealing how the perspectives of ordinary people may differ from those of professional movie critics.""")
    
    st.write(""" In conclusion, by performing IDA, EDA, and data cleaning on this dataset, we gained valuable insights into how we can apply data science theory to a real-world dataset.
              Our work is not finished yet, and we plan to update our movie app to not only work with larger datasets but also to add new features. One of these features is an update that
              will allow us to predict the movie genre, certificate, IMDb score, Metascore, and finally gross revenue by developing a machine learning model based on movie reviews. With this,
              large movie companies can simply input a review into our algorithm and predict whether it is a good choice to pursue a specific plot offered by a writer. To conclude, we will 
             show a word cloud generated from the 'overview' column in our dataset, which provides a summary of each movie. This plot illustrates the initial idea for the update to our movie app.""")

    # Visualization 7


    # Load dataset
    df = pd.read_csv('imdb_1000_final.csv')

    # Generate Word Cloud
    text = " ".join(df['Overview'].dropna().tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt.gcf())


    
    st.markdown("""
        <div style="text-align: center; font-size: 40px; font-weight: bold; font-family: 'Comic Sans MS', cursive, sans-serif; color: #ff6347;">
            Stay tuned for the final version of the movie app!!
        </div>
        """, unsafe_allow_html=True)





