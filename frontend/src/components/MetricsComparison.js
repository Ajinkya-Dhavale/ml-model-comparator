export default function MetricsComparison({ results }) {
  const metrics = [
    { label: "MSE", key: "mse" },
    { label: "R² Score", key: "r2" },
    { label: "Training Time (s)", key: "time" },
  ]

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Model Comparison</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {metrics.map((metric) => (
          <div key={metric.key} className="p-4 border rounded-lg">
            <h3 className="font-medium mb-2">{metric.label}</h3>
            <div className="space-y-2">
              <p>Linear Regression: {results["Linear Regression"][metric.key].toFixed(4)}</p>
              <p>Random Forest: {results["Random Forest"][metric.key].toFixed(4)}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
