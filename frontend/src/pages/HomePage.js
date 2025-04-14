import { Link } from "react-router-dom";
import "./HomePage.css"; // Keep your animations here

const HomePage = () => {
  const models = [
    {
      title: "Linear Regression",
      path: "/theory",
      points: [
        "Used for predicting continuous values.",
        "Equation: ŷ = β₀ + β₁x (Simple LR)",
        "Minimizes error using Least Squares.",
        "Sensitive to outliers."
      ]
    },
    {
      title: "Random Forest",
      path: "/theory",
      points: [
        "Ensemble of Decision Trees.",
        "Works by majority voting or averaging.",
        "Reduces overfitting & improves accuracy.",
        "Great for classification and regression."
      ]
    },
    {
      title: "Logistic Regression",
      path: "/theory",
      points: [
        "Used for binary classification.",
        "Formula: P(y=1) = 1 / (1 + e^-(β₀ + β₁x))",
        "Outputs probabilities between 0 and 1.",
        "Trained using Maximum Likelihood."
      ]
    }
  ];
  

  return (
    <div className="container py-4 fade-in">
      {/* Banner */}
      <div
        className="banner text-white mb-5 rounded position-relative zoom-in"
        style={{
          backgroundImage: `url('https://images.unsplash.com/photo-1639149888905-cd7a1e034c5f?auto=format&fit=crop&w=1200&q=80')`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          padding: "100px 50px"
        }}
      >
        <div
          className="position-absolute top-0 start-0 w-100 h-100"
          style={{ backgroundColor: "rgba(0,0,0,0.5)", borderRadius: "10px" }}
        ></div>
        <div className="position-relative z-1">
          <h1 className="display-4 slide-up">Machine Learning Model Comparison</h1>
          <p className="lead slide-up delay-1">
            Compare different ML algorithms, understand their strengths and weaknesses, and make informed decisions for your data science projects.
          </p>
          <Link to="/linear-vs-random-forest" className="btn btn-primary mt-3 slide-up delay-2">Start Comparing</Link>
        </div>
      </div>

      {/* Model Cards */}
      <h2 className="mb-4 slide-up">Explore Machine Learning Models</h2>
      <div className="row mb-5">
        {models.map((model, index) => (
          <div className="col-md-4 mb-4" key={index}>
            <div className={`card h-100 shadow-sm animated-card animated-delay-${index}`}>
              <div className="card-body d-flex flex-column">
                <h5 className="card-title">{model.title}</h5>
                <ul className="text-muted small mb-3" style={{ paddingLeft: "1.2rem" }}>
                  {model.points.map((point, i) => (
                    <li key={i}>{point}</li>
                  ))}
                </ul>
                <Link to={model.path} className="btn btn-primary mt-auto">Learn More</Link>
              </div>
            </div>
          </div>
        ))}
      </div>

      <hr className="my-5" />

      {/* Features */}
      <h2 className="mb-4 slide-up">What You Can Do</h2>
      <div className="row text-center">
        <div className="col-md-4 mb-4 fade-in delay-1">
          <div className="p-3">
            <div className="display-4 text-primary mb-2">🔁</div>
            <h5>Compare Models</h5>
            <p>
              See how different algorithms perform on the same dataset. Compare accuracy, precision, and training time.
            </p>
          </div>
        </div>
        <div className="col-md-4 mb-4 fade-in delay-2">
          <div className="p-3">
            <div className="display-4 text-primary mb-2">📘</div>
            <h5>Learn Theory</h5>
            <p>
              Understand the math and practical applications behind various machine learning algorithms.
            </p>
          </div>
        </div>
        <div className="col-md-4 mb-4 fade-in delay-3">
          <div className="p-3">
            <div className="display-4 text-primary mb-2">🤖</div>
            <h5>Make Predictions</h5>
            <p>
              Use trained models to make predictions on new data and compare algorithms in real-time.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
