import { useState } from "react"
import { Outlet, Link, useLocation } from "react-router-dom"
import "./Layout.css"

export default function Layout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [isDarkMode, setIsDarkMode] = useState(false)
  const location = useLocation()

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen)
  const toggleTheme = () => setIsDarkMode(!isDarkMode)

  return (
    <div className={`layout-container ${isDarkMode ? "dark-mode" : "light-mode"}`}>
      {/* Sidebar */}
      {isSidebarOpen && (
        <aside className="sidebar shadow">
          <div className="sidebar-header">
            <h4>⚙️ ML Comparator</h4>
          </div>
          <ul className="nav-links">
            <li className={location.pathname === "/" ? "active" : ""}>
              <Link to="/">🏠 Home</Link>
            </li>
            <li className={location.pathname.includes("linear-vs-random-forest") ? "active" : ""}>
              <Link to="/linear-vs-random-forest">📊 Linear vs Random Forest</Link>
            </li>
            <li className={location.pathname.includes("random-forest-vs-logistic") ? "active" : ""}>
              <Link to="/random-forest-vs-logistic">📈 Random Forest vs Logistic</Link>
            </li>
            <li className={location.pathname.includes("theory") ? "active" : ""}>
              <Link to="/theory">📘 ML Theory</Link>
            </li>
          </ul>
        </aside>
      )}

      {/* Main Content */}
      <div className="main-content">
        <nav className="navbar shadow-sm">
          <button className="btn toggle-btn" onClick={toggleSidebar}>☰</button>
          <span className="navbar-title">ML Model Comparator</span>
          <button className="btn toggle-mode" onClick={toggleTheme}>
            {isDarkMode ? "🌞 Light" : "🌙 Dark"}
          </button>
        </nav>

        <div className="content-wrapper">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
