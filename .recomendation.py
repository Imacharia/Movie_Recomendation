import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
from dash import html import dash_html_components as html
import pandas as pd
import pickle

# Load movie data
with open('movies.pkl', 'rb') as f:
    movies_data = pickle.load(f)

with open('.similarity.pkl', 'rb') as f:
    similarity_data = pickle.load(f)

with open('hybrid_recommendations.pkl', 'rb') as f:
    hybrid_recommendations_data = pickle.load(f)

with open('recommend.pkl', 'rb') as f:
    recommend_data = pickle.load(f)

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Movie Recommendation App"),
        html.Label("Select a movie:"),
        dcc.Dropdown(
            id="movie-dropdown",
            options=[
                {"label": movie_title, "value": movie_id}
                for movie_id, movie_title in movies_data["title"].items()
            ],
        ),
        html.Div(id="recommendation-output")
    ]
)

@app.callback(
    Output("recommendation-output", "children"),
    [Input("movie-dropdown", "value")]
)
def recommend_movies(movie_id):
    if movie_id is None:
        return html.Div()

    # Get recommendations for the selected movie
    recommendations = recommend_data[movie_id]

    # Get movie details for the recommendations
    movie_titles = [movies_data["title"][movie_id] for movie_id in recommendations]

    return html.Div(
        [
            html.H2("Recommended Movies"),
            html.Ul([html.Li(title) for title in movie_titles])
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True)

