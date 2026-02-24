import pandas as pd

INPUT_PATH = r"C:\Users\sragh\Documents\Multilingual_Sentiment_Project\Dataset\indian_ride_hailing_services_analysis.csv"
OUTPUT_PATH = r"C:\Users\sragh\Documents\Multilingual_Sentiment_Project\Dataset\indian_ride_hailing_services_analysis.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

print("\nColumns found:")
print(list(df.columns))

def find_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

text_col = find_column(df.columns.str.lower(), "Review")
rating_col = find_column(df.columns.str.lower(), "Rating")

# fallback: fuzzy search
if text_col is None:
    for col in df.columns:
        if "review" in col.lower() or "text" in col.lower():
            text_col = col
            break

if rating_col is None:
    for col in df.columns:
        if "rating" in col.lower() or "score" in col.lower():
            rating_col = col
            break

print(f"\nDetected text column: {text_col}")
print(f"Detected rating column: {rating_col}")

if text_col is None or rating_col is None:
    raise ValueError(
        "Could not automatically detect required columns.\n"
        "Please check your dataset manually."
    )


clean_df = df[[text_col, rating_col]].copy()
clean_df.columns = ["review", "rating"]

# remove nulls
clean_df = clean_df.dropna()

# ensure numeric rating
clean_df["rating"] = pd.to_numeric(clean_df["rating"], errors="coerce")
clean_df = clean_df.dropna()

print("\nFinal dataset shape:", clean_df.shape)

# save
clean_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nClean dataset saved to: {OUTPUT_PATH}")