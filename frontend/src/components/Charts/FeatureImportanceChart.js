"use client"

import { Bar } from "react-chartjs-2"
import { useContext } from "react"
import { DarkModeContext } from "../../context/DarkModeContext"

export default function FeatureImportanceChart({ results }) {
  const { darkMode } = useContext(DarkModeContext)

  if (!results?.["Random Forest"]?.feature_importance) return null

  const features = Object.keys(results["Random Forest"].feature_importance)
  const importance = Object.values(results["Random Forest"].feature_importance)

  // Sort features by importance
  const sortedIndices = [...Array(features.length).keys()].sort((a, b) => importance[b] - importance[a])

  const data = {
    labels: sortedIndices.map((i) => features[i]),
    datasets: [
      {
        label: "Feature Importance",
        data: sortedIndices.map((i) => importance[i]),
        backgroundColor: darkMode ? "rgba(75, 192, 192, 0.6)" : "rgba(54, 162, 235, 0.6)",
        borderColor: darkMode ? "rgba(75, 192, 192, 1)" : "rgba(54, 162, 235, 1)",
        borderWidth: 1,
      },
    ],
  }

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: "Random Forest Feature Importance",
        color: darkMode ? "white" : "black",
      },
    },
    scales: {
      x: { ticks: { color: darkMode ? "white" : "black" } },
      y: { ticks: { color: darkMode ? "white" : "black" } },
    },
  }

  return <Bar data={data} options={options} />
}
