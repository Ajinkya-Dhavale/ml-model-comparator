"use client"

import { useState } from "react"

export default function ModelControls({ columns, onTrain }) {
  const [selectedFeatures, setSelectedFeatures] = useState([])
  const [target, setTarget] = useState("")

  const handleTrain = () => {
    if (selectedFeatures.length === 0 || !target) {
      alert("Please select features and target variable")
      return
    }
    onTrain({ features: selectedFeatures, target })
  }

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md mb-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium mb-2">Features</label>
          <select
            multiple
            className="w-full border rounded-md p-2 h-32"
            onChange={(e) => setSelectedFeatures([...e.target.selectedOptions].map((o) => o.value))}
          >
            {columns.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Target Variable</label>
          <select className="w-full border rounded-md p-2" value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="">Select target</option>
            {columns.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>
      </div>

      <button onClick={handleTrain} className="mt-4 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-md">
        Train Models
      </button>
    </div>
  )
}
