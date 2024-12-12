import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    return data

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", [
    "1. Overview",
    "2. Data Exploration and Preparation",
    "3. Analysis and Insights",
    "4. Conclusions and Recommendations"
])

# Load data
data = load_data()

if section == "1. Overview":
    # App title and description
    st.title("Wine Quality Analysis")
    st.markdown("""
    This Streamlit app explores the Wine Quality dataset, performs data analysis using clustering and regression techniques, 
    and provides interactive visualizations for insights.
    """)

    # Display dataset structure
    st.subheader("Dataset Overview")
    st.dataframe(data.head())
    st.markdown("**Dataset Structure:**")
    st.write(data.info())
    st.markdown("**Descriptive Statistics:**")
    st.write(data.describe())

if section == "2. Data Exploration and Preparation":
    # Data Cleaning and Preparation
    st.title("Data Exploration and Preparation")
    st.subheader("Data Cleaning")
    if st.checkbox("Show missing values count"):
        st.write(data.isnull().sum())

    st.subheader("Correlation Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

if section == "3. Analysis and Insights":
    # Clustering Analysis
    st.title("Analysis and Insights")
    st.subheader("Clustering Analysis")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.drop("quality", axis=1))

    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data_scaled)
    st.write("Cluster Centers:")
    st.write(kmeans.cluster_centers_)

    # Cluster Visualization
    st.subheader("Cluster Visualization")
    x_col = st.selectbox("Select X-axis feature", data.columns[:-1])
    y_col = st.selectbox("Select Y-axis feature", data.columns[:-1])

    fig, ax = plt.subplots()
    sns.scatterplot(x=data[x_col], y=data[y_col], hue=data['Cluster'], palette='viridis', ax=ax)
    st.pyplot(fig)

if section == "4. Conclusions and Recommendations":
    # Conclusions and Recommendations
    st.title("Conclusions and Recommendations")
    
    st.markdown("### Key Takeaways")
    st.markdown("""
    - **Relationship Between Alcohol and Quality:** Wines with higher alcohol content tend to have better quality scores.
    - **Impact of Volatile Acidity:** High volatile acidity is often associated with lower quality wines.
    - **Sulphates Influence:** Wines with higher levels of sulphates are generally rated better in quality.
    """)
    
    st.markdown("### Recommendations for Winemakers")
    st.markdown("""
    - Optimize alcohol content to enhance the overall quality.
    - Reduce volatile acidity levels during production.
    - Consider adding appropriate amounts of sulphates to improve wine stability and quality.
    """)

    # Interactive exploration of insights
    st.markdown("### Explore Insights Interactively")
    
    insights_options = [
        "Alcohol vs Quality",
        "Volatile Acidity vs Quality",
        "Sulphates vs Quality"
    ]
    
    selected_insight = st.selectbox("Choose an insight to explore:", insights_options)
    
    if selected_insight == "Alcohol vs Quality":
        st.markdown("""
        **Insight:** Wines with higher alcohol levels tend to receive higher quality ratings. 
        Alcohol content is a significant factor in determining wine quality because it affects the flavor balance, mouthfeel, and aroma. 
        Higher alcohol content often indicates better fermentation processes and richer grape quality, which can contribute to enhanced flavor profiles.
        However, excessive alcohol can overpower other subtle flavors, so balance is key for high-quality wines.
        """)
        fig, ax = plt.subplots()
        sns.boxplot(x=data["quality"], y=data["alcohol"], ax=ax, palette="coolwarm")
        ax.set_title("Alcohol vs Wine Quality")
        ax.set_xlabel("Wine Quality")
        ax.set_ylabel("Alcohol Content")
        st.pyplot(fig)

    elif selected_insight == "Volatile Acidity vs Quality":
        st.markdown("""
        **Insight:** Higher volatile acidity is associated with lower quality ratings.
        Volatile acidity refers to the presence of acetic acid and related compounds in wine, which can result in a vinegar-like taste when present in high concentrations.
        While some level of volatile acidity is naturally present and can enhance complexity, excessive amounts indicate poor fermentation or spoilage, leading to undesirable sensory characteristics.
        Winemakers need to monitor and control volatile acidity levels to ensure the wine's overall balance and appeal.
        """)
        fig, ax = plt.subplots()
        sns.boxplot(x=data["quality"], y=data["volatile acidity"], ax=ax, palette="coolwarm")
        ax.set_title("Volatile Acidity vs Wine Quality")
        ax.set_xlabel("Wine Quality")
        ax.set_ylabel("Volatile Acidity")
        st.pyplot(fig)

    elif selected_insight == "Sulphates vs Quality":
        st.markdown("""
        **Insight:** Wines with higher sulphates levels tend to have better quality ratings. 
        Sulphates act as preservatives and antioxidants in wine, helping to maintain freshness and stability over time. 
        They play a crucial role in preventing oxidation and microbial spoilage, both of which can negatively impact wine quality.
        However, excessive sulphates can lead to a harsh taste, so careful measurement is essential. 
        This finding underscores the importance of sulphates in producing stable, high-quality wines while maintaining consumer health guidelines.
        """)
        fig, ax = plt.subplots()
        sns.boxplot(x=data["quality"], y=data["sulphates"], ax=ax, palette="coolwarm")
        ax.set_title("Sulphates vs Wine Quality")
        ax.set_xlabel("Wine Quality")
        ax.set_ylabel("Sulphates")
        st.pyplot(fig) 


