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
import { Bar, Line } from "react-chartjs-2"
import "chart.js/auto"
import CombinedDragDrop from "../components/CombinedDragDrop"
import DataPreview from "../components/DataPreview"
import PredictionForm from "../components/PredictionForm"

const LinearVsRandomForest = () => {
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
  const [irisTarget, setIrisTarget] = useState("sepal length (cm)")
  const [irisFeatures, setIrisFeatures] = useState([
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
    setTargetColumn("")
    try {
      setLoading(true)
      const response = await axios.get("http://localhost:5000/compare", {
        params: {
          features: irisFeatures.join(","), // Convert array to comma-separated string
          target: irisTarget,
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
    downloadAnchor.setAttribute("download", "regression_model_comparison_report.json")
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

  const displayTargetColumn = targetColumn === '' ? irisTarget : targetColumn;

  // Prepare Chart.js data if results exist.
  let barChartData, lineData, featureBarData
  if (results) {
    barChartData = {
      labels: ["MSE", "R2 Score", "Training Time"],
      datasets: [
        {
          label: "Linear Regression",
          data: [
            results["Linear Regression"].MSE,
            results["Linear Regression"]["R2 Score"],
            results["Linear Regression"]["Training Time"],
          ],
          backgroundColor: "rgba(75, 192, 192, 0.6)",
        },
        {
          label: "Random Forest",
          data: [
            results["Random Forest"].MSE,
            results["Random Forest"]["R2 Score"],
            results["Random Forest"]["Training Time"],
          ],
          backgroundColor: "rgba(153, 102, 255, 0.6)",
        },
      ],
    }

    if (results.Predictions && results.Predictions.Actual) {
      lineData = {
        labels: results.Predictions.Actual.map((_, i) => i + 1),
        datasets: [
          {
            label: "Actual",
            data: results.Predictions.Actual,
            borderColor: "rgba(0,0,0,0.6)",
            fill: false,
          },
          {
            label: "Linear Regression",
            data: results.Predictions["Linear Regression"],
            borderColor: "rgba(75, 192, 192, 0.6)",
            fill: false,
          },
          {
            label: "Random Forest",
            data: results.Predictions["Random Forest"],
            borderColor: "rgba(153, 102, 255, 0.6)",
            fill: false,
          },
        ],
      }
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
        Linear Regression vs Random Forest Regression
      </Typography>
      <Typography variant="body1" paragraph>
        Compare the performance of Linear Regression and Random Forest models on regression tasks. These
        algorithms have different approaches to predicting continuous values.
      </Typography>

      <Grid container spacing={4}>
        {/* Left column */}
        <Grid item xs={12} md={6} sx={{width: '100%', margin: 'auto' }} >
          {/* Iris Dataset Comparison Section */}
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Iris Dataset Regression
            </Typography>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Compare regression models on the Iris dataset to predict flower measurements.
              </Typography>
              <FormControl sx={{ minWidth: 200, mb: 2 }}>
                <InputLabel id="iris-target-label">Target Feature</InputLabel>
                <Select
                  labelId="iris-target-label"
                  value={irisTarget}
                  label="Target Feature"
                  onChange={(e) => setIrisTarget(e.target.value)}
                >
                  {irisCols.map((col) => (
                    <MenuItem key={col} value={col}>
                      {col}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl sx={{ minWidth: 300,marginLeft: 2 }}>
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
                    <MenuItem key={col} value={col} disabled={col === irisTarget}>
                      <Checkbox checked={irisFeatures.indexOf(col) > -1} />
                      <ListItemText primary={col} />
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
            <Button variant="contained" onClick={handleCompareIris} disabled={loading}>
              Compare Regression Models
            </Button>
          </Paper>

          {/* Custom File Upload */}
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Upload Your Data
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Upload your own CSV file to evaluate regression models on your data. The file should contain numerical data for 
              regression analysis.
            </Typography>
            <CombinedDragDrop onFileSelect={handleFileSelect} />
          </Paper>

          {/* CSV Preview & Selection Controls for Custom Dataset */}
          {columns.length > 0 && (
            <Paper sx={{ p: 3, mb: 3 ,maxWidth:"970px", overflowX:"auto"}}>
              <Typography variant="h6" gutterBottom>
                CSV Preview
              </Typography>
              <Box sx={{  maxHeight: 310, overflow: "auto", mb: 2 }}>
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
            Regression Model Evaluation Results
          </Typography>

          <Grid container spacing={3} sx={{ mb: 4 }}>
            {Object.entries(results).map(([model, metrics]) => {
              if (model === "Predictions") return null
              return (
                <Grid item xs={12} md={6} key={model} sx={{ p: 2, width:"48%" }}>
                  <Paper sx={{ p: 2, height: "100%", display: "flex", flexDirection: "column" }}>
                    <Typography variant="h6" gutterBottom>
                      {model}
                    </Typography>
                    {Object.entries(metrics).map(([metric, value]) => {
                      if (metric === "Feature Importance") return null
                      return (
                        <Typography variant="body2" key={metric}>
                          {metric}: {typeof value === "number" ? value.toFixed(4) : String(value)}
                        </Typography>
                      )
                    })}
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
  <Grid item xs={12} md={6}>
    <Paper sx={{ p: 2, height: 300, width:450 }}>
      <Typography variant="subtitle1" gutterBottom>
        Performance Metrics
      </Typography>
      <Bar
        data={barChartData}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: "top",
            },
          },
        }}
      />
    </Paper>
  </Grid>

  {lineData && (
    <Grid item xs={12} md={6}>
      <Paper sx={{ p: 2, height: 300, width:450 }}>
        <Typography variant="subtitle1" gutterBottom>
          Prediction Trends (Test Set)
        </Typography>
        <Line
          data={lineData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                display: true,
                position: "top",
              },
            },
          }}
        />
      </Paper>
    </Grid>
  )}
</Grid>


{/* Prediction Form */}
{predictionSchema.length > 0 && (
            <Paper sx={{ p: 3, mb: 3, marginTop: 4 }}>
              <Typography variant="h6" gutterBottom>
                Make Regression Predictions
              </Typography>
              <PredictionForm
                schema={predictionSchema}
                values={predictionValues}
                onChange={handlePredictionInputChange}
                onPredict={handlePredict}
                result={predictionResult}
                isClassification={false}
                targetColumn={displayTargetColumn}
              />
            </Paper>
          )}

{/* Right column */}
<Grid item xs={12} md={6} sx={{ marginTop: 4 }}>



          {/* Model Comparison Theory */}
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Regression Algorithm Comparison
            </Typography>
            <Typography variant="body1" sx={{ mb: 2 }}>
              <strong>Linear Regression:</strong> A statistical model that estimates the relationship between variables by 
              fitting a linear equation to observed data. It assumes a linear relationship between the input variables and 
              the target variable.
            </Typography>
            <Typography variant="body1" sx={{ mb: 2 }}>
              <strong>Random Forest Regression:</strong> An ensemble learning method that builds multiple decision trees during 
              training and outputs the average prediction of the individual trees. It can capture non-linear relationships.
            </Typography>
            <Typography variant="body1" component="div" sx={{ mb: 2 }}>
              <strong>Key Differences:</strong>
              <ul>
                <li>
                  <strong>Model Complexity:</strong> Linear Regression creates a simple linear function, while Random Forest 
                  builds many complex tree models.
                </li>
                <li>
                  <strong>Interpretability:</strong> Linear Regression provides easily interpretable coefficients, 
                  while Random Forest is more of a black box model.
                </li>
                <li>
                  <strong>Performance on Complex Data:</strong> Random Forest often outperforms Linear Regression on 
                  datasets with non-linear relationships.
                </li>
                <li>
                  <strong>Overfitting Risk:</strong> Linear Regression is less prone to overfitting on simple datasets, 
                  while Random Forest has built-in protection against overfitting through ensemble methods.
                </li>
              </ul>
            </Typography>
            <Typography variant="h6" gutterBottom>
              Evaluation Metrics
            </Typography>
            <Typography variant="body1" component="div" sx={{ mb: 2 }}>
              <ul>
                <li>
                  <strong>Mean Squared Error (MSE):</strong> Measures the average squared difference between predicted and actual values. 
                  Lower values indicate better performance.
                </li>
                <li>
                  <strong>R² Score:</strong> Represents the proportion of variance in the dependent variable that is predictable 
                  from the independent variables. Values closer to 1 indicate better fit.
                </li>
                <li>
                  <strong>Training Time:</strong> The time required to train the model. Faster training allows for more 
                  experimentation and iteration.
                </li>
              </ul>
            </Typography>
            <Typography variant="h6" gutterBottom>
              Learning Resources
            </Typography>
            <Typography variant="body1" sx={{ mb: 1 }}>
              <strong>Linear Regression:</strong>{" "}
              <a href="https://www.youtube.com/watch?v=nk2CQITm_eo" target="_blank" rel="noopener noreferrer">
                Watch Video Tutorial
              </a>
            </Typography>
            <Typography variant="body1">
              <strong>Random Forest:</strong>{" "}
              <a href="https://www.youtube.com/watch?v=eM4uJ6XGnSM" target="_blank" rel="noopener noreferrer">
                Watch Video Tutorial
              </a>
            </Typography>
          </Paper>

          
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

export default LinearVsRandomForest