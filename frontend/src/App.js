// App.js
import { BrowserRouter, Routes, Route } from "react-router-dom"
import HomePage from "./pages/HomePage"
import LinearVsRandomForest from "./pages/LinearVsRandomForest"
import RandomForestVsLogistic from "./pages/RandomForestVsLogistic"
import TheoryPage from "./pages/TheoryPage"
import Layout from "./Layout"

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="linear-vs-random-forest" element={<LinearVsRandomForest />} />
          <Route path="random-forest-vs-logistic" element={<RandomForestVsLogistic />} />
          <Route path="theory" element={<TheoryPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
