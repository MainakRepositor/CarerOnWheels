import streamlit as st
import pandas as pd
import sqlite3
import random
import numpy as np
from scipy.spatial.distance import cdist
from itertools import permutations
import matplotlib.pyplot as plt
import heapq

# Create a database connection
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Create patient and carer tables if not exist
cursor.execute('''CREATE TABLE IF NOT EXISTS patients
                (name TEXT, symptom TEXT, location TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS carers
                (name TEXT, address TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS distances
                (patient_location TEXT, carer_location TEXT, distance REAL)''')
conn.commit()

# Function to add data to database
def add_patient_data(name, symptom, location):
    cursor.execute("INSERT INTO patients (name, symptom, location) VALUES (?, ?, ?)", (name, symptom, location))
    conn.commit()

def add_carer_data(name, address):
    cursor.execute("INSERT INTO carers (name, address) VALUES (?, ?)", (name, address))
    conn.commit()

def add_distance(patient_location, carer_location, distance):
    cursor.execute("INSERT INTO distances (patient_location, carer_location, distance) VALUES (?, ?, ?)", 
                   (patient_location, carer_location, distance))
    conn.commit()

# Function to download data as CSV
def download_table_as_csv(table_name):
    data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    csv = data.to_csv(index=False)
    return csv

# Function to clear the database
def clear_database():
    cursor.execute("DELETE FROM patients")
    cursor.execute("DELETE FROM carers")
    cursor.execute("DELETE FROM distances")
    conn.commit()

# Function to fetch data from a table
def fetch_table_data(table_name):
    data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    return data

# Travelling Salesman Problem Solver
def tsp_solver(locations, distances):
    shortest_path = None
    min_distance = float('inf')
    for perm in permutations(locations):
        current_distance = sum(distances[perm[i], perm[i+1]] for i in range(len(perm) - 1))
        if current_distance < min_distance:
            min_distance = current_distance
            shortest_path = perm
    return shortest_path, min_distance

# Homepage
def homepage():
    st.title("Welcome to the Carer On Wheels")
    st.image("careronwheels.jpeg", use_column_width=True)
    st.markdown('''The patient care application system is proposed to optimise assigning caretakers to patients, ensuring efficient and personalized care delivery. This app defines how the carer can travel in the shortest time to the patients and treat them ensuring fastest care''')

# Data Entry Page
def data_entry():
    st.title("Data Entry Page")

    # Patient Data Entry
    st.header("Enter Patient Data")
    patient_name = st.text_input("Patient Name")
    symptom = st.text_input("Symptom")
    location = st.text_input("Location")
    if st.button("Add Patient"):
        add_patient_data(patient_name, symptom, location)
        st.success("Patient data added successfully.")

    # Carer Data Entry
    st.header("Enter Carer Data")
    carer_name = st.text_input("Carer Name")
    carer_address = st.text_input("Carer Address")
    if st.button("Add Carer"):
        add_carer_data(carer_name, carer_address)
        st.success("Carer data added successfully.")

    # Adding distances between patients and carers
    st.header("Add Distance Between Patients and Carer")
    patient_location = st.text_input("Patient Location")
    carer_location = st.text_input("Carer Location")
    distance = st.number_input("Distance (in km)", min_value=0.0)
    if st.button("Add Distance"):
        add_distance(patient_location, carer_location, distance)
        st.success("Distance added successfully.")

    # Download Data as CSV
    st.header("Download Data as CSV")
    if st.button("Download Patient Data"):
        csv = download_table_as_csv('patients')
        st.download_button("Download CSV", csv, "patients_data.csv")
    
    if st.button("Download Carer Data"):
        csv = download_table_as_csv('carers')
        st.download_button("Download CSV", csv, "carers_data.csv")

    # Clear Database
    st.header("Clear Database")
    if st.button("Clear All Data"):
        clear_database()
        st.success("All data cleared successfully.")

    # View Data
    st.header("View Data")
    table_name = st.selectbox("Select Table to View", ["patients", "carers", "distances"])
    if st.button("View Table Data"):
        data = fetch_table_data(table_name)
        st.write(f"Data from table: {table_name}")
        st.dataframe(data)

# A* Pathfinding Algorithm
def astar_path(grid, start, end):
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, end), 0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    
    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < rows) and (0 <= neighbor[1] < cols) and grid[neighbor] == 0:
                tentative_g_score = g_score[current] + np.linalg.norm(np.array(current) - np.array(neighbor))
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    if neighbor not in [i[2] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))
    
    return []

def mapper_page():
    st.title("Mapper Page")
    
    # Creating a 64x64 grid
    grid_size = 64
    grid = np.zeros((grid_size, grid_size))

    # Add obstacles randomly
    obstacles = [(random.randint(0, grid_size-1), random.randint(0, grid_size-1)) for _ in range(100)]
    for obstacle in obstacles:
        grid[obstacle] = 1

    # Displaying grid
    st.write("64x64 Grid (1s are obstacles):")
    st.image(grid, width=500)

    # Fetch patient locations from database
    patients = pd.read_sql_query("SELECT name, location FROM patients", conn)
    carers = pd.read_sql_query("SELECT name, address FROM carers", conn)

    if patients.empty or carers.empty:
        st.warning("No patients or carers available to map.")
        return

    # Randomly place patients on the grid
    patient_locations = [(random.randint(0, grid_size-1), random.randint(0, grid_size-1)) for _ in range(len(patients))]
    
    # Assign carer
    carer_location = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
    st.write(f"Assigned Carer Location: {carer_location}")

    # Plot patients and carers on the grid
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray')

    # Plot patients
    for i, loc in enumerate(patient_locations):
        ax.text(loc[1], loc[0], patients['name'][i], color="red", fontsize=12)

    # Plot carer
    ax.text(carer_location[1], carer_location[0], "Carer", color="blue", fontsize=12)

    # Calculate TSP for shortest path
    locations = [carer_location] + patient_locations
    distances = cdist(locations, locations, metric='euclidean')
    path, min_dist = tsp_solver(list(range(len(locations))), distances)

    # Display shortest path
    st.write(f"Shortest Path: {path}")
    st.write(f"Minimum Distance: {min_dist:.2f} km")

    # Display nearest 5 patients
    st.write("Nearest 5 Patients:")
    nearest_patients = np.argsort(distances[0])[1:6]
    for i in nearest_patients:
        st.write(patients['name'][i-1])  # Minus 1 because index 0 is the carer

    # Draw green lines to nearest 5 patients with pathfinding
    sorted_patients = [patient_locations[i-1] for i in nearest_patients]
    current_location = carer_location
    
    for patient_location in sorted_patients:
        path = astar_path(grid, current_location, patient_location)
        if path:
            for i in range(len(path) - 1):
                ax.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]], color='green', linewidth=2)
        current_location = patient_location

    st.pyplot(fig)

# Streamlit Page Navigation
page = st.sidebar.selectbox("Select Page", ["Homepage", "Data Entry", "Mapper"])

if page == "Homepage":
    homepage()
elif page == "Data Entry":
    data_entry()
elif page == "Mapper":
    mapper_page()
