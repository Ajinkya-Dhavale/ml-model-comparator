import { Grid, Paper, Typography } from "@mui/material"

const ModelComparison = ({ results }) => {
  return (
    <Grid container spacing={3} sx={{ mb: 4 }}>
      {Object.entries(results).map(([model, metrics]) => {
        if (model === "Predictions" || model === "Confusion Matrix" || model === "Classification Report") return null
        return (
          <Grid item xs={12} md={6} key={model}>
            <Paper sx={{ p: 2, height: "100%", display: "flex", flexDirection: "column" }}>
              <Typography variant="h6" gutterBottom>
                {model}
              </Typography>
              {Object.entries(metrics).map(([metric, value]) => (
                <Typography variant="body2" key={metric}>
                  {metric}: {typeof value === "number" ? value.toFixed(4) : value}
                </Typography>
              ))}
            </Paper>
          </Grid>
        )
      })}
    </Grid>
  )
}

export default ModelComparison
