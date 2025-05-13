import json
import argparse
import pandas as pd

def rank_ntu_labels_by_density(json_path):
    with open(json_path, 'r') as f:
        stats = json.load(f)

    if "labels" not in stats:
        raise ValueError("Expected top-level key 'labels' in the JSON file.")

    labels_data = stats["labels"]

    # Build DataFrame
    data = [
        {
            "rank": None,
            "action_id": idx,
            "action_label": item["action_label"],
            "density_score": item["density_score"]
        }
        for idx, item in enumerate(labels_data)
    ]
    df = pd.DataFrame(data)
    df = df.sort_values(by="density_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Clean and aligned print
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.colheader_justify", "center")
    pd.set_option("display.float_format", "{:,.6f}".format)

    print("\n📊 NTU Labels ranked by estimated density:\n")
    print(df[["rank", "action_id", "action_label", "density_score"]].to_string(index=False))

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-path", required=True, type=str,
        help="Path to JSON file with output of 'compute_knn_stats'"
    )
    args = parser.parse_args()

    df_full = rank_ntu_labels_by_density(args.json_path)
