import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math

# Load the transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

model = load_model()

# App title
st.title("🔍 Course Transferability Checker")
st.markdown("Compare course descriptions and titles to estimate how likely one course will transfer to another.")

# Input fields
st.subheader("📘 Course Descriptions")
text1a = st.text_area("Description 1", placeholder="Enter first course description...", height=80)
text1b = st.text_area("Description 2", placeholder="Enter second course description...", height=80)

st.subheader("📗 Course Titles")
text2a = st.text_area("Title 1", placeholder="Enter first course title...", height=80)
text2b = st.text_area("Title 2", placeholder="Enter second course title...", height=80)

# Compare button
if st.button("🔍 Compare & Check Transferability Score"):
    if not all([text1a.strip(), text1b.strip(), text2a.strip(), text2b.strip()]):
        st.error("❌ Please fill in all four fields.")
    else:
        try:
            # Encode and compute similarities
            embeddings_a = model.encode([text1a, text1b])
            similarity_a = cosine_similarity([embeddings_a[0]], [embeddings_a[1]])[0][0]

            embeddings_b = model.encode([text2a, text2b])
            similarity_b = cosine_similarity([embeddings_b[0]], [embeddings_b[1]])[0][0]

            # Logistic regression formula (custom-trained)
            combined_score = 1 / (1 + math.exp(-(-13.969 + 15.533 * similarity_a + 8.048 * similarity_b)))

            # Display results
            st.success("✅ Transferability Analysis Complete!")

            st.markdown("### 📊 Similarity Scores")
            st.markdown(f"**Descriptions Similarity:** `{similarity_a:.4f}` ({similarity_a * 100:.2f}%)")
            st.markdown(f"**Titles Similarity:** `{similarity_b:.4f}` ({similarity_b * 100:.2f}%)")

            st.markdown("### 🎯 Combined Transferability Score")
            st.markdown(f"**Transfer Likelihood:** `{combined_score:.4f}` ({combined_score * 100:.2f}%)")

            # Interpretation
            if combined_score >= 0.8:
                interpretation = "🟢 Very High Transferability"
            elif combined_score >= 0.6:
                interpretation = "🔵 High Transferability"
            elif combined_score >= 0.4:
                interpretation = "🟡 Moderate Transferability"
            elif combined_score >= 0.2:
                interpretation = "🟠 Low Transferability"
            else:
                interpretation = "🔴 Very Low Transferability"

            st.markdown(f"**Interpretation:** {interpretation}")

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
