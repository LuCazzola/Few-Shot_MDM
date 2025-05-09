import json
import argparse
import pandas as pd

def rank_ntu_labels_by_density(json_path, top_n=None):
    with open(json_path, 'r') as f:
        stats = json.load(f)

    # Flatten if wrapped
    if "stats_per_label" in stats:
        stats = stats["stats_per_label"]

    # Build DataFrame
    data = [
        {
            "rank": None,
            "action_id": int(action_id),
            "action_label": info["action_label"],
            "density_score": info["density_score"]
        }
        for action_id, info in stats.items()
    ]
    df = pd.DataFrame(data)
    df = df.sort_values(by="density_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    if top_n:
        df = df.head(top_n)

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
    parser.add_argument("json_path", type=str, help="Path to JSON file with density scores")
    parser.add_argument("--top", type=int, default=None, help="Show only top N entries")
    args = parser.parse_args()

    df_full = rank_ntu_labels_by_density(args.json_path, args.top)
