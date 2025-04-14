"use client"

import { useState, useEffect } from "react"
import axios from "axios"
import {
  Box,
  Button,
  CircularProgress,
  Container,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Snackbar,
  Typography,
  Checkbox,
  ListItemText,
  OutlinedInput,
} from "@mui/material"
import { Alert } from "@mui/material"
import { Bar } from "react-chartjs-2"
import "chart.js/auto"
import CombinedDragDrop from "../components/CombinedDragDrop"
import DataPreview from "../components/DataPreview"
import PredictionForm from "../components/PredictionForm"
import ConfusionMatrix from "../components/ConfusionMatrix"

const RandomForestVsLogistic = () => {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Custom dataset upload states
  const [uploadedFile, setUploadedFile] = useState(null)
  const [columns, setColumns] = useState([])
  const [previewData, setPreviewData] = useState([])
  const [targetColumn, setTargetColumn] = useState("")
  const [selectedFeatures, setSelectedFeatures] = useState([])

  // Prediction state for custom dataset
  const [predictionSchema, setPredictionSchema] = useState([])
  const [predictionValues, setPredictionValues] = useState({})
  const [predictionResult, setPredictionResult] = useState(null)

  // Iris dataset comparison states
  const irisCols = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
  const [irisFeatures, setIrisFeatures] = useState([
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
  ])

  // When a new custom file is uploaded, clear previous selections.
  useEffect(() => {
    setSelectedFeatures([])
    setPredictionSchema([])
    setPredictionValues({})
  }, [uploadedFile])

  // Ensure custom dataset target is removed from selectedFeatures.
  useEffect(() => {
    setSelectedFeatures((prev) => prev.filter((col) => col !== targetColumn))
  }, [targetColumn])

  // Handle custom file selection via CombinedDragDrop.
  const handleFileSelect = (file) => {
    if (!file) return
    setUploadedFile(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      const text = e.target.result
      const lines = text.split("\n").filter((line) => line.trim() !== "")
      if (lines.length === 0) return
      const header = lines[0].split(",")
      setColumns(header)
      // Default: last column as target.
      setTargetColumn(header[header.length - 1])
      // Auto-select remaining columns as features.
      setSelectedFeatures(header.slice(0, header.length - 1))
      const preview = lines.slice(1, 6).map((line) => {
        const values = line.split(",")
        const obj = {}
        header.forEach((col, index) => {
          obj[col] = values[index]
        })
        return obj
      })
      setPreviewData(preview)
    }
    reader.readAsText(file)
  }

  // Handler for Iris dataset comparison.
  const handleCompareIris = async () => {
    try {
      setLoading(true)
      // Use the regular /compare endpoint but pass species as target
      const response = await axios.get("http://localhost:5000/compare", {
        params: {
          features: irisFeatures.join(","), // Convert array to comma-separated string
          target: "species",
        },
      })

      if (response.data.success) {
        setResults(response.data.results)
        if (response.data.prediction_schema) {
          setPredictionSchema(response.data.prediction_schema)
          const initVals = {}
          response.data.prediction_schema.forEach((item) => {
            initVals[item.name] = ""
          })
          setPredictionValues(initVals)
        }
        setError(null)
      } else {
        setError(response.data.error || "Failed to compare models")
      }
    } catch (err) {
      console.error("API Error:", err)
      setError(err.response?.data?.error || "Failed to compare models")
    } finally {
      setLoading(false)
    }
  }

  // Handler for custom dataset training.
  const handleCustomTrain = async () => {
    if (!uploadedFile) return
    try {
      setLoading(true)
      const formData = new FormData()
      formData.append("file", uploadedFile)
      formData.append("target", targetColumn)
      // Send only the selected independent features.
      selectedFeatures.forEach((feature) => formData.append("features", feature))
      // Use the regular /upload endpoint - the backend will detect if it's classification
      const response = await axios.post("http://localhost:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      setResults(response.data.results)
      if (response.data.prediction_schema) {
        setPredictionSchema(response.data.prediction_schema)
        const initVals = {}
        response.data.prediction_schema.forEach((item) => {
          initVals[item.name] = ""
        })
        setPredictionValues(initVals)
      }
      setError(null)
    } catch (err) {
      setError(err.response?.data?.error || "File upload failed")
    } finally {
      setLoading(false)
    }
  }

  // Download report.
  const handleDownload = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(results, null, 2))
    const downloadAnchor = document.createElement("a")
    downloadAnchor.setAttribute("href", dataStr)
    downloadAnchor.setAttribute("download", "classification_model_comparison_report.json")
    document.body.appendChild(downloadAnchor)
    downloadAnchor.click()
    downloadAnchor.remove()
  }

  // Handle changes in prediction input fields.
  const handlePredictionInputChange = (feature, value) => {
    setPredictionValues((prev) => ({ ...prev, [feature]: value }))
  }

  // Send prediction request.
  const handlePredict = async () => {
    try {
      // Use the regular /predict endpoint - the backend knows if it's classification
      const response = await axios.post("http://localhost:5000/predict", { features: predictionValues })
      if (response.data.success) {
        setPredictionResult(response.data.predictions)
        setError(null)
      } else {
        setError(response.data.error || "Prediction failed")
      }
    } catch (err) {
      setError(err.response?.data?.error || "Prediction failed")
    }
  }

  // Prepare Chart.js data if results exist.
  let barChartData, featureBarData
  if (results) {
    // Adjust the metrics based on what's available in the results
    const metrics = ["Accuracy", "Training Time"]
    if (results["Logistic Regression"]["Precision"] !== undefined) {
      metrics.splice(1, 0, "Precision", "Recall", "F1 Score")
    }

    barChartData = {
      labels: metrics,
      datasets: [
        {
          label: "Logistic Regression",
          data: metrics.map((metric) => results["Logistic Regression"][metric]),
          backgroundColor: "rgba(255, 99, 132, 0.6)",
        },
        {
          label: "Random Forest",
          data: metrics.map((metric) => results["Random Forest"][metric]),
          backgroundColor: "rgba(153, 102, 255, 0.6)",
        },
      ],
    }

    if (results["Random Forest"].hasOwnProperty("Feature Importance")) {
      const importance = results["Random Forest"]["Feature Importance"]
      featureBarData = {
        labels: Object.keys(importance),
        datasets: [
          {
            label: "Feature Importance",
            data: Object.values(importance),
            backgroundColor: "rgba(255, 159, 64, 0.6)",
          },
        ],
      }
    }
  }

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Random Forest vs Logistic Regression
      </Typography>
      <Typography variant="body1" paragraph>
        Compare the performance of Random Forest and Logistic Regression models on classification tasks. These
        algorithms have different approaches to classification problems.
      </Typography>

      <Grid container spacing={4}>
        {/* Left column */}
        <Grid item xs={12} md={6} sx={{ width:"100%" }}>
          {/* Iris Dataset Comparison Section */}
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Iris Dataset Classification
            </Typography>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Compare classification models on the Iris dataset to predict flower species.
              </Typography>
              <FormControl sx={{ minWidth: 300 }}>
                <InputLabel id="iris-features-label">Features to Use</InputLabel>
                <Select
                  labelId="iris-features-label"
                  multiple
                  value={irisFeatures}
                  onChange={(e) => setIrisFeatures(e.target.value)}
                  input={<OutlinedInput label="Features to Use" />}
                  renderValue={(selected) => selected.join(", ")}
                >
                  {irisCols.map((col) => (
                    <MenuItem key={col} value={col}>
                      <Checkbox checked={irisFeatures.indexOf(col) > -1} />
                      <ListItemText primary={col} />
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
            <Button variant="contained" onClick={handleCompareIris} disabled={loading}>
              Compare Classification Models
            </Button>
          </Paper>

         

          {/* CSV Preview & Selection Controls for Custom Dataset */}
          {columns.length > 0 && (
            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                CSV Preview
              </Typography>
              <Box sx={{ maxHeight: 300, overflow: "auto", mb: 2 }}>
                <DataPreview columns={columns} data={previewData} />
              </Box>
              <Box sx={{ mt: 2, display: "flex", gap: 2, flexWrap: "wrap" }}>
                <FormControl sx={{ minWidth: 150 }}>
                  <InputLabel id="target-select-label">Target Column</InputLabel>
                  <Select
                    labelId="target-select-label"
                    value={targetColumn}
                    label="Target Column"
                    onChange={(e) => setTargetColumn(e.target.value)}
                  >
                    {columns.map((col) => (
                      <MenuItem key={col} value={col}>
                        {col}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl sx={{ minWidth: 250 }}>
                  <InputLabel id="features-select-label">Independent Features</InputLabel>
                  <Select
                    labelId="features-select-label"
                    multiple
                    value={selectedFeatures}
                    onChange={(e) => setSelectedFeatures(e.target.value)}
                    input={<OutlinedInput label="Independent Features" />}
                    renderValue={(selected) => selected.join(", ")}
                  >
                    {columns
                      .filter((col) => col !== targetColumn)
                      .map((col) => (
                        <MenuItem key={col} value={col}>
                          <Checkbox checked={selectedFeatures.indexOf(col) > -1} />
                          <ListItemText primary={col} />
                        </MenuItem>
                      ))}
                  </Select>
                </FormControl>
              </Box>
              <Box sx={{ mt: 2 }}>
                <Button variant="contained" onClick={handleCustomTrain} disabled={loading}>
                  Evaluate Custom Dataset
                </Button>
              </Box>
            </Paper>
          )}
        </Grid>

       
      </Grid>

      {loading && (
        <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {/* Results Section */}
      {results && (
        <Paper sx={{ p: 3, my: 3 }}>
          <Typography variant="h5" gutterBottom>
            Classification Model Evaluation Results
          </Typography>

          <Grid container spacing={3} sx={{ mb: 4 }} >
            {Object.entries(results).map(([model, metrics]) => {
              if (model === "Predictions" || model === "Confusion Matrix" || model === "Classification Report")
                return null
              return (
                <Grid item xs={12} md={6} key={model}  sx={{ p: 2, width:"48%" }}>
                  <Paper sx={{ p: 2, height: "100%", display: "flex", flexDirection: "column" }}>
                    <Typography variant="h6" gutterBottom>
                      {model}
                    </Typography>
                    {Object.entries(metrics).map(([metric, value]) => (
                      <Typography variant="body2" key={metric}>
                        {metric}: {typeof value === "number" ? value.toFixed(4) : String(value)}
                      </Typography>
                    ))}
                  </Paper>
                </Grid>
              )
            })}
          </Grid>

          <Divider sx={{ my: 4 }} />

          <Typography variant="h6" gutterBottom>
            Graphical Comparison
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Performance Metrics
                </Typography>
                <Bar data={barChartData} />
              </Paper>
            </Grid>

            {results["Confusion Matrix"] && (
              <Grid item xs={12}  sx={{width:"100%" }}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Confusion Matrices
                  </Typography>
                  <Grid container spacing={2} >
                    <Grid item xs={12} md={6} sx={{ width:"48%" }}>
                      <Typography variant="subtitle2" align="center">
                        Logistic Regression
                      </Typography>
                      <ConfusionMatrix data={results["Confusion Matrix"]["Logistic Regression"]} />
                    </Grid>
                    <Grid item xs={12} md={6}  sx={{ width:"42%" }}>
                      <Typography variant="subtitle2" align="center">
                        Random Forest
                      </Typography>
                      <ConfusionMatrix data={results["Confusion Matrix"]["Random Forest"]} />
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>
            )}

            <Grid item xs={12} sx={{ width:"100%" }}>
               {/* Prediction Form */}
          {predictionSchema.length > 0 && (
            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Make Classification Predictions
              </Typography>
              <PredictionForm
                schema={predictionSchema}
                values={predictionValues}
                onChange={handlePredictionInputChange}
                onPredict={handlePredict}
                result={predictionResult}
                isClassification={true}
                targetColumn= "Species Type"
              />
            </Paper>
          )}
            </Grid>

 {/* Right column */}
 <Grid item xs={12} md={6}>
          {/* Model Comparison Theory */}
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Classification Algorithm Comparison
            </Typography>
            <Typography variant="body1" sx={{ mb: 2 }}>
              <strong>Logistic Regression:</strong> A statistical model that uses a logistic function to model a binary
              dependent variable. It's widely used for classification problems and provides probability scores.
            </Typography>
            <Typography variant="body1" sx={{ mb: 2 }}>
              <strong>Random Forest Classifier:</strong> An ensemble method that builds multiple decision trees and uses
              majority voting for classification. It can handle both binary and multi-class problems.
            </Typography>
            <Typography variant="body1" component="div" sx={{ mb: 2 }}>
              <strong>Key Differences:</strong>
              <ul>
                <li>
                  <strong>Decision Boundary:</strong> Logistic Regression creates a linear decision boundary, while
                  Random Forest can create complex non-linear boundaries.
                </li>
                <li>
                  <strong>Interpretability:</strong> Logistic Regression provides odds ratios that are easy to
                  interpret, while Random Forest is more of a black box.
                </li>
                <li>
                  <strong>Performance:</strong> Random Forest often performs better on complex datasets but requires
                  more computational resources.
                </li>
                <li>
                  <strong>Robustness:</strong> Random Forest is less affected by outliers and requires less feature
                  engineering.
                </li>
              </ul>
            </Typography>
            <Typography variant="h6" gutterBottom>
              Learning Resources
            </Typography>
            <Typography variant="body1" sx={{ mb: 1 }}>
              <strong>Logistic Regression:</strong>{" "}
              <a href="https://www.youtube.com/watch?v=yIYKR4sgzI8" target="_blank" rel="noopener noreferrer">
                Watch Video Tutorial
              </a>
            </Typography>
            <Typography variant="body1">
              <strong>Random Forest:</strong>{" "}
              <a href="https://www.youtube.com/watch?v=J4Wdy0Wc_xQ" target="_blank" rel="noopener noreferrer">
                Watch Video Tutorial
              </a>
            </Typography>
          </Paper>

         
        </Grid>

          </Grid>

          <Box sx={{ mt: 3, display: "flex", justifyContent: "flex-end" }}>
            <Button variant="outlined" onClick={handleDownload}>
              Download Report
            </Button>
          </Box>
        </Paper>
      )}

      <Snackbar open={!!error} autoHideDuration={6000} onClose={() => setError(null)}>
        <Alert severity="error" sx={{ width: "100%" }}>
          {error}
        </Alert>
      </Snackbar>
    </Container>
  )
}

export default RandomForestVsLogistic
