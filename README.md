## 📝 PDF Document Section Classifier

**Python Version:** 3.8+  
**License:** MIT  
**Model Size:** ≤ 15MB  
**Execution:** CPU-only  
**Internet Requirement:** None (fully offline)

---

### 📌 Project Overview

This project processes and classifies sections of PDF documents into structural categories like `Title`, `H1`, `H2`, `H3`, and `Other`.  
It uses feature-based supervised learning and works completely offline without requiring internet access.

---

### 🧠 Approach Summary

1. **PDF Text Block Extraction**  
   Text blocks and their visual features (e.g., font size, boldness, spacing) are extracted using `pdfplumber`.

2. **Manual Labeling**  
   Extracted blocks are labeled manually into five categories:  
   - `Title`  
   - `H1`  
   - `H2`  
   - `H3`  
   - `Other`

3. **Model Training**  
   A `DecisionTreeClassifier` is trained using these features:  
   - Font size  
   - Boldness  
   - Line spacing above  
   - Prefix patterns (e.g., "1.", "1.2.")  
   - Capitalization  
   - Line length

4. **Prediction on Unseen PDFs**  
   For new PDFs, the trained model predicts the label of each block using the same feature extraction logic.

5. **Structured JSON Output**  
   Predicted blocks are exported into a structured JSON format representing the hierarchy of sections.

---

### ✅ Key Features

- 📴 No internet connection required  
- ⚡ Lightweight and fast inference  
- 🛠️ Easy to adapt to other document styles and work on multilingual documents 
- 📄 Output in clean JSON format for downstream use
