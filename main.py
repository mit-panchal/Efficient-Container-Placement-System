import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import heapq
from datetime import datetime

# Load data from CSV files
yard_locations = pd.read_csv("Yard Locations.csv")
past_in_out_data = pd.read_csv("Past In and Out Container Data.csv")
incoming_containers = pd.read_csv("Incoming Conatiners.csv")

# Preprocess data
yard_locations["Location"] = (
    yard_locations["Area"].astype(str)
    + yard_locations["Row"].astype(str)
    + yard_locations["Bay"].astype(str)
    + yard_locations["Level"].astype(str)
)

past_in_out_data["IN_TIME"] = pd.to_datetime(
    past_in_out_data["IN_TIME"], format="%d-%m-%Y %H:%M", errors="coerce"
)
past_in_out_data["OUT_TIME"] = pd.to_datetime(
    past_in_out_data["OUT_TIME"], format="%d-%m-%Y %H:%M", errors="coerce"
)
incoming_containers["IN_TIME"] = pd.to_datetime(incoming_containers["IN_TIME"])

past_in_out_data.dropna(subset=["IN_TIME", "OUT_TIME"], inplace=True)
incoming_containers.dropna(subset=["IN_TIME"], inplace=True)

# Train Linear Regression model with data preprocessing
imputer = SimpleImputer(strategy="mean")
X = past_in_out_data[["CON_SIZE"]]
X_imputed = imputer.fit_transform(X)
model = LinearRegression()
y = (past_in_out_data["OUT_TIME"] - datetime(1970, 1, 1)).dt.total_seconds()
model.fit(X_imputed, y)

# Initialize data structures
priority_queue = []
available_space = {
    loc: {"Container Size": 0, "Location Status": "empty"}
    for loc in yard_locations["Location"]
}
assigned_locations = {}

# Populate priority queue
for index, container in incoming_containers.iterrows():
    predicted_departure_time = model.predict([[container["CON_SIZE"]]])[0]
    heapq.heappush(priority_queue, (predicted_departure_time, container))

# Placement algorithm
while priority_queue:
    predicted_departure_time, container = heapq.heappop(priority_queue)
    optimal_location = find_optimal_location(container, available_space)
    assigned_locations[container["ID"]] = optimal_location
    update_available_space(optimal_location, container["CON_SIZE"], available_space)

# Save assigned locations to CSV
resultant_data = [
    {"ID": container_id, "Assigned Location": location}
    for container_id, location in assigned_locations.items()
]
resultant = pd.DataFrame(resultant_data)
resultant.to_csv("ResultTab.csv", index=False)
