import pandas as pd

#reads JSON file and converts it to CSV for specific sample dataset we're using
#df = pd.read_json("Cell_Phones_and_Accessories_5.json", lines=True)
#df.to_csv("Cell_Phones_and_Accessories_5.csv", index=False)
def convert_json_to_csv(input_path: str, output_path: str) -> None:
    df = pd.read_json(input_path, lines=True)
    df.to_csv(output_path, index=False)

    print(f" Conversion complete! Saved as {output_path}")