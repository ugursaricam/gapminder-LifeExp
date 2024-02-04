import streamlit as st
import plotly.express as px
import joblib
import pandas as pd

@st.cache_data
def get_data():
    df = px.data.gapminder()
    return df

def get_model():
    model = joblib.load("gapminder.joblib")
    return model

st.title("Gapminder - Life Expectancy")
main_page, data_page, tab_vis, model_page = st.tabs(["Main Page", "Dataset", "Visualizations", "ML Model"])


main_page.image("images/image1.png")
main_page.subheader("Introduction:")
main_page.markdown("""
The Gapminder dataset is an extensive compilation of global development data that spans a wide array of socio-economic indicators across numerous countries over several decades. It encapsulates key metrics that are indicative of the health, economic conditions, and overall well-being of populations worldwide. This dataset includes vital statistics such as country names, categorized by continent, and detailed yearly records of life expectancy, population size, and GDP per capita. These elements allow for a nuanced analysis of global development trends, offering insights into the economic growth and health outcomes across different nations and the evolving dynamics over time. Serving as an invaluable resource for educators, policymakers, researchers, and anyone with a keen interest in global development matters, the Gapminder dataset's rich combination of economic, demographic, and health indicators provides a powerful means to examine the various factors that impact the quality of life in various parts of the globe.
""")

main_page.subheader("How it Works:")
main_page.markdown("""
- Data Loading: The application starts by loading the Gapminder dataset using Plotly's px.data.gapminder() function, which contains comprehensive socio-economic indicators across various countries and decades.

- Feature Selection: For the analysis, the application focuses on three main features: year, representing the time dimension; pop, indicating the population size of each country; and gdpPercap, which stands for GDP per capita. The target variable for prediction is lifeExp, the life expectancy at birth.
- Data Splitting: The dataset is divided into training and testing sets, with 80% of the data allocated for training the models and 20% reserved for testing their performance. This split ensures that the model can be evaluated on unseen data, providing an accurate measure of its predictive capabilities.
- Model Training: Multiple regression models, including Linear Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost, are trained on the Gapminder data. The application calculates the Root Mean Square Error (RMSE) for each model using 5-fold cross-validation, allowing for a robust comparison of their predictive accuracy.
- Voting Regressor: The application employs a Voting Regressor that combines predictions from the best-performing models. This ensemble method leverages the strengths of individual models to improve overall prediction accuracy on life expectancy.
- Model Evaluation: The Voting Regressor's performance is evaluated using cross-validation, and its RMSE is displayed, offering insights into the ensemble's effectiveness in predicting life expectancy.
- Model Saving: Once trained, the final model is saved as a .joblib file, making it easy to deploy or share with others.""")

df = get_data()
data_page.dataframe(df, width=900)

data_page.subheader("Key Features:")
data_page.markdown("""
- Country: This field identifies the name of the country to which the data row pertains. It includes a wide range of nations from various continents, offering a global perspective on development trends.
- Continent: The continent field categorizes each country into one of several continental regions, such as Africa, Americas, Asia, Europe, and Oceania. This classification helps in regional analysis and comparisons.
- Year: Data is provided for multiple years, allowing for time-series analyses. This temporal aspect of the dataset enables researchers and analysts to track changes and development progress over time.
- Life Expectancy (lifeExp): This metric represents the average number of years a newborn is expected to live if current mortality rates continue to apply. It's a critical indicator of public health and overall living conditions.
- Population (pop): The population field shows the total number of people living in a country in a given year. It provides insight into the demographic size and growth trends of nations.
- GDP per Capita (gdpPercap): This indicator measures the average economic output per person, adjusted for inflation to current U.S. dollars. It's a widely used gauge of economic prosperity and living standards.
""")


tab_vis.subheader("Comparison of Life Expectancy of Selected Countries by Years")

selected_countries = tab_vis.multiselect(label="Ülke Seçiniz", options=df.country.unique(), default=["Turkey", "Syria", "Greece"])
filtered_df = df[df.country.isin(selected_countries)]

fig = px.line(
    filtered_df,
    x="year",
    y="lifeExp",
    color="country"
)

tab_vis.plotly_chart(fig, use_container_width=True)


tab_vis.subheader("Showing the Change in Life Expectancy of Countries Over the Years on a Map")
year_select_for_map = tab_vis.slider("Years ", min_value=int(df.year.min()), max_value=int(df.year.max()),
                                     step=5)

fig2 = px.choropleth(df[df.year == year_select_for_map], locations="iso_alpha",
                     color="lifeExp",
                     range_color=(df.lifeExp.min(), df.lifeExp.max()),
                     hover_name="country",
                     color_continuous_scale=px.colors.sequential.Plasma)

tab_vis.plotly_chart(fig2, use_container_width=True)


tab_vis.subheader("Population, GDP and Life Expectancy Changes of Countries Over the Years")
fig3 = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent",
                  animation_group='country', animation_frame="year",
                  hover_name="country", range_x=[100, 100000], range_y=[25, 90], log_x=True, size_max=60)
fig3.add_hline(y=50, line_dash="dash", line_color="black")
tab_vis.plotly_chart(fig3, use_container_width=True)


model = get_model()

year = model_page.number_input("Year", min_value=1952, max_value=2027, step=1, value=2000)
pop = model_page.number_input("Population", min_value=10000, max_value=1000000000,  step=100000, value=1000000)
gdpPercap = model_page.number_input("GDP", min_value=1, step=1, value=5000)

user_input = pd.DataFrame({'year':year, 'pop':pop, 'gdpPercap': gdpPercap}, index=[0])

if model_page.button("Predict!"):
    prediction = model.predict(user_input)
    model_page.success(f"Predicted Life Expectancy: {prediction[0]}")