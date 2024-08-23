import re
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import os

# filepath = os.path.join(os.getcwd(), 'LLama-7B_HPD','result','HPD',"lora-checkpoint-checkpoint-1140","log")
# filename = "loraThu Aug 15 20-16-05"

filepath = os.path.join(os.getcwd(), 'LLama-7B_PIK','result','PIK',"lora-checkpoint-checkpoint-1420","log")
filename = "loraThu Aug 15 23-55-59"

# filepath = os.path.join(os.getcwd(), 'LLama-13B_PIK','result','PIK',"lora-checkpoint-checkpoint-1400","log")
# filename = "loraThu Aug 15 11-05-56"

# filepath = os.path.join(os.getcwd(), 'LLama-13B_HPD','result','HPD',"lora-checkpoint-checkpoint-920","log")
# filename = "loraThu Aug 15 11-02-10"

# Read the entire file as a single large string
with open(filepath + os.sep + filename, 'r', encoding='utf-8') as file:
    data = file.read()

# Split the content into blocks using 'Finished' as the delimiter
blocks = data.split('Finished')

results = []
labels = []

# Process each block
for block in blocks:
    if not block.strip():
        continue  # Skip empty blocks

    # Extract Result and labels
    result = re.findall(r'Result:(\d+)</s>', block)
    label = re.findall(r'labels:(\d+)', block)

    if not result:
        print(f"Warning: No Result found in the following block:\n{block}\n")
    
    if result and label:
        results.extend(result)
        labels.extend(label)

print(f"Results count: {len(results)}")
print(f"Labels count: {len(labels)}")

results = list(map(int, results))
labels = list(map(int, labels))

precision, recall, f1, _ = precision_recall_fscore_support(labels, results, average='weighted')
accuracy = accuracy_score(labels, results)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# report = classification_report(labels, results)
# print("\nClassification Report:\n")
# print(report)
