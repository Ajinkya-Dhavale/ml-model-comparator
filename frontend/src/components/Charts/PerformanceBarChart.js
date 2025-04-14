import { Bar } from "react-chartjs-2"

export default function PerformanceBarChart({ metrics }) {
  const data = {
    labels: ["MSE", "R² Score"],
    datasets: [
      {
        label: "Linear Regression",
        data: [metrics.linear.mse, metrics.linear.r2],
        backgroundColor: "rgba(255, 99, 132, 0.5)",
      },
      {
        label: "Random Forest",
        data: [metrics.rf.mse, metrics.rf.r2],
        backgroundColor: "rgba(54, 162, 235, 0.5)",
      },
    ],
  }

  return <Bar data={data} />
}
