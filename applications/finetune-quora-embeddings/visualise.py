import pandas as pd
import dataframe_image as dfi
import json

with open("results.json", "r") as f:
    data = json.load(f)

model_names = data.keys()

# Create a pandas dataframe
df = pd.DataFrame(data.values(), index=model_names)

dfi.export(df.round(4), "df_styled_non_transpose.png")
df = df.transpose().round(4)
dfi.export(df, "df_styled.png")
