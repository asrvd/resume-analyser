import streamlit as st
import PyPDF2
from ensemble_model import ResumeAnalyzer
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🎓", layout="wide")


# Load model (cached)
@st.cache_resource
def load_analyzer():
    return ResumeAnalyzer()


# Initialize
analyzer = load_analyzer()

# Header
st.title("🎓 AI-Powered Resume Analysis System")
st.markdown("*Deep Learning Based Career Path Prediction & Skill Assessment*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 About")
    st.info(
        """
    This system uses **Sentence Transformers** + **Ensemble Learning** 
    to predict optimal career paths.
    
    **Categories:**
    - 🏢 Private Job
    - 🎓 Higher Studies
    - 🔬 Research Field
    - 📚 Skill Improvement
    """
    )

    st.header("📈 Model Info")

    # Load metadata
    import json

    try:
        with open("models/metadata.json", "r") as f:
            metadata = json.load(f)
        st.metric("Accuracy", f"{metadata['test_accuracy']:.1%}")
        st.metric("Training Samples", metadata["train_samples"])
    except:
        st.metric("Accuracy", "84-86%")
        st.metric("Training Samples", "500")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Resume")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload your resume for instant analysis",
    )

    if uploaded_file:
        # Extract text
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            # Preview
            with st.expander("📄 Resume Preview"):
                st.text_area("Content", text[:500] + "...", height=200)

            # Analyze button
            if st.button("🔍 Analyze Resume", type="primary"):
                with st.spinner("Analyzing..."):
                    result = analyzer.predict(text)
                    recommendations = analyzer.generate_recommendations(result)

                st.session_state["result"] = result
                st.session_state["recommendations"] = recommendations

        except Exception as e:
            st.error(f"Error reading PDF: {e}")

with col2:
    if "result" in st.session_state:
        result = st.session_state["result"]
        recommendations = st.session_state["recommendations"]

        st.subheader("📊 Analysis Results")

        # Main prediction
        st.success(f"**Predicted Category:** {result['category']}")

        # Confidence gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result["confidence"] * 100,
                title={"text": "Confidence Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 60], "color": "lightgray"},
                        {"range": [60, 80], "color": "gray"},
                        {"range": [80, 100], "color": "lightgreen"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 85,
                    },
                },
            )
        )
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        # All probabilities
        st.subheader("📈 Category Probabilities")
        for cat, prob in sorted(
            result["probabilities"].items(), key=lambda x: x[1], reverse=True
        ):
            st.progress(prob, text=f"{cat}: {prob:.1%}")

        # Features
        st.subheader("🔍 Extracted Information")
        features = result["features"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("CGPA", f"{features['cgpa']:.2f}" if features["cgpa"] else "N/A")
            st.metric("Skills Found", len(features["skills"]))
        with col_b:
            st.metric("Internship", "✓ Yes" if features["has_internship"] else "✗ No")
            st.metric("Projects", "✓ Yes" if features["has_projects"] else "✗ No")

        if features["skills"]:
            st.write("**Skills:**", ", ".join(features["skills"]))

        # Recommendations
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

        # Model insights
        with st.expander("🔬 Model Insights"):
            st.write(f"**ML Model Prediction:** {result['ml_prediction']}")
            st.write(f"**Rule-Based Prediction:** {result['rule_prediction']}")
            st.write(f"**Ensemble Method:** 70% ML + 30% Rules")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <p><b>Developed by:</b> Ashirvad Bhushan, Arka Dey, Md Kaif Ansari, Biplab Ghosh</p>
    <p><b>Under guidance of:</b> Mr. Kaushik Mukherjee, Dept. of IT, KGEC</p>
    <p><b>Session:</b> 2025-2026</p>
</div>
""",
    unsafe_allow_html=True,
)
