"use client"

import { Scatter } from "react-chartjs-2"
import { useContext } from "react"
import { DarkModeContext } from "../../context/DarkModeContext"

export default function ScatterPlot({ results }) {
  const { darkMode } = useContext(DarkModeContext)

  if (!results?.predictions) return null

  const { actual, lr_pred, rf_pred } = results.predictions

  const data = {
    datasets: [
      {
        label: "Linear Regression",
        data: actual.map((val, i) => ({ x: val, y: lr_pred[i] })),
        backgroundColor: darkMode ? "rgba(255, 99, 132, 0.6)" : "rgba(255, 99, 132, 0.6)",
      },
      {
        label: "Random Forest",
        data: actual.map((val, i) => ({ x: val, y: rf_pred[i] })),
        backgroundColor: darkMode ? "rgba(54, 162, 235, 0.6)" : "rgba(54, 162, 235, 0.6)",
      },
    ],
  }

  const options = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: "Actual vs Predicted Values",
        color: darkMode ? "white" : "black",
      },
      legend: {
        labels: { color: darkMode ? "white" : "black" },
      },
    },
    scales: {
      x: {
        title: { display: true, text: "Actual Values", color: darkMode ? "white" : "black" },
        ticks: { color: darkMode ? "white" : "black" },
      },
      y: {
        title: { display: true, text: "Predicted Values", color: darkMode ? "white" : "black" },
        ticks: { color: darkMode ? "white" : "black" },
      },
    },
  }

  return <Scatter data={data} options={options} />
}
