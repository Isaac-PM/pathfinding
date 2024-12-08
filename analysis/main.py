import streamlit as st
import pandas as pd

DATA_PATH = "../build/results.csv"

CURRENT_ROUND_SESSION_KEY = "current_round"

PLAYER1_CSV_KEY = "Player 1"
PLAYER2_CSV_KEY = "Player 2"
ROUND_CSV_COLUMN = "Round"
PLAYER_CSV_COLUMN = "Player"
POINTS_CSV_COLUMN = "Points"
WON_ROUND_CSV_COLUMN = "Player Won Round"
TIME_IN_ROUND_CSV_COLUMN = "Time (milliseconds)"


@st.cache_data
def load_data():
    csv_path = DATA_PATH
    data = pd.read_csv(csv_path, sep=",")
    return data

def transform_to_vertical_headers(data):
    transformed_data = data.to_frame().reset_index()
    transformed_data.columns = ["Attribute", "Value"]
    return transformed_data

if CURRENT_ROUND_SESSION_KEY not in st.session_state:
    st.session_state[CURRENT_ROUND_SESSION_KEY] = 1

data = load_data()
round_count = data[ROUND_CSV_COLUMN].max()
current_data = data[data[ROUND_CSV_COLUMN] == st.session_state[CURRENT_ROUND_SESSION_KEY]]
player1_data = current_data[current_data[PLAYER_CSV_COLUMN] == PLAYER1_CSV_KEY].iloc[0]
player2_data = current_data[current_data[PLAYER_CSV_COLUMN] == PLAYER2_CSV_KEY].iloc[0]

total_points_player1 =  data[data[ROUND_CSV_COLUMN] == round_count][data[PLAYER_CSV_COLUMN] == PLAYER1_CSV_KEY][POINTS_CSV_COLUMN].sum()
total_points_player2 =  data[data[ROUND_CSV_COLUMN] == round_count][data[PLAYER_CSV_COLUMN] == PLAYER2_CSV_KEY][POINTS_CSV_COLUMN].sum()
total_points_winner = PLAYER1_CSV_KEY if total_points_player1 > total_points_player2 else PLAYER2_CSV_KEY
total_points_winner = "Draw" if total_points_player1 == total_points_player2 else total_points_winner 

total_won_rounds_player1 = data[data[PLAYER_CSV_COLUMN] == PLAYER1_CSV_KEY][WON_ROUND_CSV_COLUMN].sum()
total_won_rounds_player2 = data[data[PLAYER_CSV_COLUMN] == PLAYER2_CSV_KEY][WON_ROUND_CSV_COLUMN].sum()
total_won_rounds_winner = PLAYER1_CSV_KEY if total_won_rounds_player1 > total_won_rounds_player2 else PLAYER2_CSV_KEY
total_won_rounds_winner = "Draw" if total_won_rounds_player1 == total_won_rounds_player2 else total_won_rounds_winner

efficiency_player1 = total_points_player1 / data[data[PLAYER_CSV_COLUMN] == PLAYER1_CSV_KEY][TIME_IN_ROUND_CSV_COLUMN].sum()
efficiency_player2 = total_points_player2 / data[data[PLAYER_CSV_COLUMN] == PLAYER2_CSV_KEY][TIME_IN_ROUND_CSV_COLUMN].sum()
efficiency_winner = PLAYER1_CSV_KEY if efficiency_player1 > efficiency_player2 else PLAYER2_CSV_KEY
efficiency_winner = "Draw" if efficiency_player1 == efficiency_player2 else efficiency_winner

st.title("Pathfinding Results")
st.markdown("For more information refer to the README.md file, remember to run the game before trying to display the results with this page.")
st.divider()

def previous_round():
    if st.session_state[CURRENT_ROUND_SESSION_KEY] > 1:
        st.session_state[CURRENT_ROUND_SESSION_KEY] -= 1

def next_round():
    if st.session_state[CURRENT_ROUND_SESSION_KEY] + 1 <= round_count:
        st.session_state[CURRENT_ROUND_SESSION_KEY] += 1
    else:
        st.session_state[CURRENT_ROUND_SESSION_KEY] = 1

st.header("Game Results")
results_table = pd.DataFrame([
    {"Player": total_points_winner, "Winning Criteria": "Won by points (highest point count)"},
    {"Player": total_won_rounds_winner, "Winning Criteria": "Won by rounds (highest won rounds)"},
    {"Player": efficiency_winner, "Winning Criteria": "Won by efficiency (highest total points over total time)"}
])
st.table(results_table)


previous_button, next_button = st.columns(2)
with previous_button:
    if st.button("Previous Round", on_click=previous_round):
        pass
with next_button:
    if st.button("Next Round", on_click=next_round):
        pass

player1_column, player2_column = st.columns(2)

with player1_column:
    st.header("Player 1")
    st.subheader("Solution")
    st.image(f"../build/path_player1_round_{st.session_state[CURRENT_ROUND_SESSION_KEY]}.ppm", caption="Player 1 solution")
    st.subheader("Round Information")
    st.table(transform_to_vertical_headers(player1_data))
    
with player2_column:
    st.header("Player 2")
    st.subheader("Solution")
    st.image(f"../build/path_player2_round_{st.session_state[CURRENT_ROUND_SESSION_KEY]}.ppm", caption="Player 2 solution")
    st.subheader("Round Information")
    st.table(transform_to_vertical_headers(player2_data))