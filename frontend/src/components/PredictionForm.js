"use client"
import {
  Box,
  Button,
  TextField,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Typography,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material"

const PredictionForm = ({ schema, values, onChange, onPredict, result, isClassification = false, targetColumn }) => {
  return (
    <Box>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {schema.map((item) => (
          <Box key={item.name}>
            {item.type === "numeric" ? (
              <TextField
                label={item.name}
                value={values[item.name] || ""}
                onChange={(e) => onChange(item.name, e.target.value)}
                type="number"
                fullWidth
                size="small"
              />
            ) : (
              <FormControl fullWidth size="small">
                <InputLabel id={`${item.name}-select-label`}>{item.name}</InputLabel>
                <Select
                  labelId={`${item.name}-select-label`}
                  value={values[item.name] || ""}
                  label={item.name}
                  onChange={(e) => onChange(item.name, e.target.value)}
                >
                  {item.options &&
                    item.options.map((option) => (
                      <MenuItem key={option} value={option}>
                        {option}
                      </MenuItem>
                    ))}
                </Select>
              </FormControl>
            )}
          </Box>
        ))}

        <Button variant="contained" onClick={onPredict} sx={{ mt: 1 }}>
          Predict
        </Button>
      </Box>

      {result && (
        <Box sx={{ mt: 4 }}>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Prediction Results for <strong>{targetColumn}</strong>
          </Typography>

          <TableContainer component={Paper} sx={{ mt: 2 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Model</strong></TableCell>
                  <TableCell><strong>Prediction</strong></TableCell>
                  {isClassification && <TableCell><strong>Probability</strong></TableCell>}
                </TableRow>
              </TableHead>
              <TableBody>
                {isClassification ? (
                  <>
                    <TableRow>
                      <TableCell>Logistic Regression</TableCell>
                      <TableCell>{String(result["Logistic Regression"])}</TableCell>
                      <TableCell>
                        {result["Logistic Regression Probability"]
                          ? `${(result["Logistic Regression Probability"] * 100).toFixed(2)}%`
                          : "—"}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Random Forest</TableCell>
                      <TableCell>{String(result["Random Forest"])}</TableCell>
                      <TableCell>
                        {result["Random Forest Probability"]
                          ? `${(result["Random Forest Probability"] * 100).toFixed(2)}%`
                          : "—"}
                      </TableCell>
                    </TableRow>
                  </>
                ) : (
                  <>
                    <TableRow>
                      <TableCell>Linear Regression</TableCell>
                      <TableCell>{String(result["Linear Regression"])}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Random Forest</TableCell>
                      <TableCell>{String(result["Random Forest"])}</TableCell>
                    </TableRow>
                  </>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}
    </Box>
  )
}

export default PredictionForm
