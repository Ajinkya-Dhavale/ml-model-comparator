import { Box, Typography } from "@mui/material"

const ConfusionMatrix = ({ data }) => {
  // Check if data exists
  if (!data) {
    return (
      <Box sx={{ textAlign: "center", p: 2 }}>
        <Typography variant="body2">Confusion matrix data not available</Typography>
      </Box>
    )
  }

  // Extract class labels and matrix values
  const classes = Object.keys(data)

  return (
    <Box sx={{ display: "flex", justifyContent: "center", mt: 2 }}>
      <Box sx={{ display: "inline-block" }}>
        {/* Header row with predicted labels */}
        <Box sx={{ display: "flex", mb: 1 }}>
          <Box sx={{ width: 100 }}></Box>
          <Typography variant="subtitle2" sx={{ textAlign: "center", fontWeight: "bold", mb: 1 }}>
            Predicted
          </Typography>
        </Box>

        <Box sx={{ display: "flex" }}>
          <Box sx={{ width: 100 }}>
            <Typography
              variant="subtitle2"
              sx={{
                textAlign: "center",
                fontWeight: "bold",
                mb: 1,
                transform: "rotate(-90deg)",
                height: 100,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              Actual
            </Typography>
          </Box>

          <Box>
            {/* Column headers */}
            <Box sx={{ display: "flex", mb: 1 }}>
              <Box sx={{ width: 30 }}></Box>
              {classes.map((cls) => (
                <Box key={cls} sx={{ width: 60, textAlign: "center" }}>
                  <Typography variant="body2">{cls}</Typography>
                </Box>
              ))}
            </Box>

            {/* Matrix rows */}
            {classes.map((actualClass) => (
              <Box key={actualClass} sx={{ display: "flex", mb: 1 }}>
                <Box sx={{ width: 30, display: "flex", alignItems: "center" }}>
                  <Typography variant="body2">{actualClass}</Typography>
                </Box>
                {classes.map((predictedClass) => {
                  // Safely access the value, providing a default of 0
                  const value =
                    data[actualClass] && data[actualClass][predictedClass] !== undefined
                      ? data[actualClass][predictedClass]
                      : 0

                  return (
                    <Box
                      key={predictedClass}
                      sx={{
                        width: 60,
                        height: 60,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        bgcolor: actualClass === predictedClass ? "rgba(76, 175, 80, 0.2)" : "rgba(244, 67, 54, 0.1)",
                        border: "1px solid",
                        borderColor: actualClass === predictedClass ? "success.main" : "error.main",
                      }}
                    >
                      <Typography variant="body1" fontWeight="bold">
                        {value}
                      </Typography>
                    </Box>
                  )
                })}
              </Box>
            ))}
          </Box>
        </Box>
      </Box>
    </Box>
  )
}

export default ConfusionMatrix
