import os
import pdfplumber
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import json

MERGED_CSV = "merged_blocks.csv"
INFERENCE_PDF = "file05.pdf"

# ---------- Feature Extraction ----------
def extract_blocks_features(pdf_path):
    blocks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            last_y0 = 0
            for page_num, page in enumerate(pdf.pages):
                for obj in page.extract_words(extra_attrs=["fontname", "size"]):
                    text = obj["text"].strip()
                    if not text:
                        continue
                    block = {
                        "text": text,
                        "font_size": obj["size"],
                        "font_name": obj["fontname"],
                        "is_bold": 1 if "Bold" in obj["fontname"] or "bold" in obj["fontname"].lower() else 0,
                        "y0": obj["top"],
                        "page_num": page_num + 1,
                        "line_length": len(text.split()),
                        "whitespace_above": obj["top"] - last_y0 if blocks else obj["top"],
                        "prefix_pattern": 1 if text.split() and text.split()[0].rstrip(".").replace(".", "").isdigit() else 0,
                        "text_case": ("ALLCAPS" if text.isupper() else
                                      "Title" if text.istitle() else
                                      "lower" if text.islower() else "mixed")
                    }
                    blocks.append(block)
                    last_y0 = obj["top"]
        return pd.DataFrame(blocks)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return pd.DataFrame()

# ---------- Training ----------
try:
    df = pd.read_csv(MERGED_CSV)
    df = pd.get_dummies(df, columns=["font_name", "text_case"])
    X = df.drop(["label", "text"], axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, random_state=42)
    dtree.fit(X_train, y_train)
    print(f"Validation Accuracy: {dtree.score(X_test, y_test):.2f}")

    with open("training_columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

except Exception as e:
    print(f"Error during training: {e}")
    exit()

# ---------- Inference ----------
try:
    if not os.path.exists(INFERENCE_PDF):
        raise FileNotFoundError(f"{INFERENCE_PDF} not found in the current directory.")

    new_df = extract_blocks_features(INFERENCE_PDF)
    if new_df.empty:
        print("Error: No data extracted from PDF.")
        exit()

    new_df = pd.get_dummies(new_df, columns=["font_name", "text_case"])

    with open("training_columns.pkl", "rb") as f:
        training_columns = pickle.load(f)

    new_X = new_df.reindex(columns=training_columns, fill_value=0)
    predicted_labels = dtree.predict(new_X)

    results = new_df[["page_num", "text"]].copy()
    results["predicted_label"] = predicted_labels

    # ---------- Build JSON ----------
    document_json = {
        "title": "",
        "outline": []
    }

    title_text = ""
    for _, row in results.iterrows():
        if row["predicted_label"] == "Title":
            title_text += row["text"] + " "
        elif title_text:
            document_json["title"] = title_text.strip()
            break
    if title_text and not document_json["title"]:
        document_json["title"] = title_text.strip()

    current_heading = ""
    current_label = ""
    current_page = None

    for _, row in results.iterrows():
        label = row["predicted_label"]

        if label in ["H1", "H2", "H3"]:
            if label != current_label:
                if current_heading:
                    document_json["outline"].append({
                        "level": current_label,
                        "text": current_heading.strip(),
                        "page": int(current_page)
                    })
                current_heading = row["text"] + " "
                current_label = label
                current_page = row["page_num"]
            else:
                current_heading += row["text"] + " "

    if current_heading:
        document_json["outline"].append({
            "level": current_label,
            "text": current_heading.strip(),
            "page": int(current_page)
        })

    with open("output_outline.json", "w", encoding="utf-8") as f:
        json.dump(document_json, f, indent=4)

    print("JSON file created: output_outline.json")

except Exception as e:
    print(f"Error during prediction: {e}")
