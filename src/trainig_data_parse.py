import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

# Initialize the parser
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",  # Outputs perfect markdown for your AI
    num_workers=4            # Makes parsing faster
)

print("Parsing your textbook... Please wait.")
# Parse the book
documents = parser.load_data("ayurveda_textbook.pdf")

# Save the clean text to a file
with open("clean_ayurveda_data.md", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text + "\n")

print("🎉 Done! Open 'clean_ayurveda_data.md' to see your perfectly formatted text.")