# app.py
import numpy as np
import streamlit as st
import joblib

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Classification")
st.write("Predict the Iris species from sepal/petal measurements.")

FEATURE_ORDER = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

DEFAULT_RANGES = {
    "sepal length (cm)": (4.3, 7.9, 5.8),
    "sepal width (cm)" : (2.0, 4.4, 3.0),
    "petal length (cm)": (1.0, 6.9, 3.8),
    "petal width (cm)" : (0.1, 2.5, 1.2),
}

@st.cache_resource
def load_model(path: str):
    try:
        model = joblib.load(path)
        return model, None
    except FileNotFoundError:
        return None, "❌ Model file 'iris_model.pkl' not found. Run your notebook to save it in the same folder as app.py."
    except Exception as e:
        return None, f"❌ Could not load model: {e}\nTip: Retrain & save the model in the same environment you're running Streamlit."

model, load_err = load_model("iris_model.pkl")

with st.sidebar:
    st.header("🔧 Inputs")
    vals = {}
    for feat in FEATURE_ORDER:
        lo, hi, default = DEFAULT_RANGES[feat]
        vals[feat] = st.slider(feat, float(lo), float(hi), float(default), step=0.1)

    predict_clicked = st.button("🔮 Predict")

if load_err:
    st.error(load_err)

elif predict_clicked:
    X = np.array([[vals[f] for f in FEATURE_ORDER]])
    try:
        pred = model.predict(X)[0]
        # try to get probabilities if available
        proba_text = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = getattr(model, "classes_", [0,1,2])
            # pretty print as table
            st.subheader("Class probabilities")
            st.dataframe(
                {"class": classes, "probability": [float(p) for p in proba]}
            )
            proba_text = " (see table below)"
        name_map = {0: "Setosa 🌱", 1: "Versicolor 🌸", 2: "Virginica 🌺"}
        st.success(f"**Prediction:** {name_map.get(int(pred), str(pred))}{proba_text}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}\nCheck feature order and model compatibility.")
else:
    st.info("Use the sliders in the sidebar and click **Predict**.")
