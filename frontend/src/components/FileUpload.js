import { useDropzone } from "react-dropzone"
import Papa from "papaparse"

export default function FileUpload({ onUpload }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: ".csv",
    multiple: false,
    onDrop: (files) => {
      Papa.parse(files[0], {
        complete: (result) => onUpload(result.data),
        header: true,
        dynamicTyping: true,
      })
    },
  })

  return (
    <div
      {...getRootProps()}
      className={`p-8 border-2 border-dashed rounded-lg mb-8 cursor-pointer 
        ${isDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300"}`}
    >
      <input {...getInputProps()} />
      <p className="text-center">{isDragActive ? "Drop CSV file here" : "Drag & drop CSV file, or click to select"}</p>
    </div>
  )
}
