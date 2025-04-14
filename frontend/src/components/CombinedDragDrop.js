"use client"

import { Box, Typography } from "@mui/material"

const CombinedDragDrop = ({ onFileSelect }) => {
  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) onFileSelect(file)
  }

  const handleDragOver = (e) => e.preventDefault()

  const handleClick = () => {
    document.getElementById("csv-upload").click()
  }

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) onFileSelect(file)
  }

  return (
    <Box
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={handleClick}
      sx={{
        border: "2px dashed",
        borderColor: "grey.500",
        borderRadius: 1,
        p: 3,
        textAlign: "center",
        cursor: "pointer",
        transition: "all 0.3s",
        "&:hover": {
          borderColor: "primary.main",
          bgcolor: "rgba(63, 81, 181, 0.04)",
        },
      }}
    >
      <Typography variant="body1">Drag and drop your CSV file here, or click to select one</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        Supported format: .csv
      </Typography>
      <input accept=".csv" id="csv-upload" type="file" style={{ display: "none" }} onChange={handleFileChange} />
    </Box>
  )
}

export default CombinedDragDrop
