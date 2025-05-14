import json
import argparse
from transformers import pipeline

# Define your prompt template
PROMPT_TEMPLATE = (
    "Expand the following short action description into 5 diverse natural language captions. "
    "Each caption should describe the same action in a different way. "
    "Be concise, use natural language, and always describe a human performing the action.\n\n"
    "Action: {action}\n"
    "Captions:"
)

def generate_captions(model, action_label):
    prompt = PROMPT_TEMPLATE.format(action=action_label)
    output = model(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]['generated_text']

    # Extract captions from the output
    # Try splitting by newlines or expected list formats
    generated = output.split("Captions:")[-1].strip().split("\n")
    captions = [cap.strip("1234567890. -") for cap in generated if cap.strip()]
    return [cap for cap in captions if cap.lower().startswith(("a ", "someone", "somebody", "person"))]

def main(json_path):
    # Load model
    print("Loading language model...")
    gen_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

    # Load input JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Augment captions
    for key, entry in data.items():
        action = entry["action_label"]
        print(f"Generating for: {action}")
        new_captions = generate_captions(gen_model, action)
        entry["captions"].extend(new_captions)

    # Save to new JSON
    out_path = json_path.replace(".json", "_augmented.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved augmented data to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to the input JSON file")
    args = parser.parse_args()
    main(args.json_file)
