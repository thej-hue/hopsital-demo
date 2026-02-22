// App.js
import React, { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import Dashboard from "./components/Dashboard";
import Patients from "./components/Patients";
import PatientDetail from "./components/PatientDetail";
import CohortAnalysis from "./components/CohortAnalysis";
import CohortSegmentation from "./components/CohortSegmentation";
import PredictiveAnalysis from "./components/PredictiveAnalysis";
import Login from "./components/Login";
import PatientPredictiveDetail from "./components/PatientPredictiveDetail";
import Register from "./components/Register";

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(
    localStorage.getItem("isLoggedIn") === "true"
  );

  const handleLogin = (email) => {
    setIsLoggedIn(true);
    localStorage.setItem("isLoggedIn", "true");
    localStorage.setItem("doctorEmail", email);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("doctorEmail");
  };

  const ProtectedRoute = ({ children }) => {
    return isLoggedIn ? children : <Navigate to="/login" replace />;
  };

  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<Login onLogin={handleLogin} />} />
        <Route path="/register" element={<Register />} />

        {/* Protected Routes */}
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Dashboard onLogout={handleLogout} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/patients"
          element={
            <ProtectedRoute>
              <Patients onLogout={handleLogout} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/patients/:id"
          element={
            <ProtectedRoute>
              <PatientDetail onLogout={handleLogout} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/cohort-segmentation"
          element={
            <ProtectedRoute>
              <CohortSegmentation onLogout={handleLogout} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/cohort-analysis"
          element={
            <ProtectedRoute>
              <CohortAnalysis onLogout={handleLogout} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/patient-predictive/:id"
          element={
            <ProtectedRoute>
              <PatientPredictiveDetail doctorEmail={localStorage.getItem("doctorEmail")} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/predictive-analysis"
          element={
            <ProtectedRoute>
              <PredictiveAnalysis
                doctorEmail={localStorage.getItem("doctorEmail")}
                onLogout={handleLogout}
              />
            </ProtectedRoute>
          }
        />

        {/* Fallback */}
        <Route
          path="*"
          element={<Navigate to={isLoggedIn ? "/" : "/login"} replace />}
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;