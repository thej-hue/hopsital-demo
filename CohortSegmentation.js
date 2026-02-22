import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import patientsData from "../data/patients.json";

export default function CohortSegmentation({ onLogout }) {
  const navigate = useNavigate();

  const [selectedFeature, setSelectedFeature] = useState("manual");
  const [ageFilter, setAgeFilter] = useState("");
  const [conditionFilter, setConditionFilter] = useState("");
  const [manualCohorts, setManualCohorts] = useState([]);
  const [showCohortNameInput, setShowCohortNameInput] = useState(false);
  const [cohortName, setCohortName] = useState("");

  // AI Cohorts (for demo)
  const aiCohorts = [
    { name: "AI Cohort 1", patients: patientsData.slice(0, 3) },
    { name: "AI Cohort 2", patients: patientsData.slice(3, 6) },
    { name: "AI Cohort 3", patients: patientsData.slice(6, 9) },
  ];

  const handleCreateManualCohort = () => {
    const filtered = patientsData.filter(
      (p) =>
        (ageFilter ? p.age >= ageFilter : true) &&
        (conditionFilter
          ? p.condition.toLowerCase().includes(conditionFilter.toLowerCase())
          : true)
    );

    if (filtered.length === 0) {
      alert("No patients match the selected filters.");
      return;
    }

    setShowCohortNameInput(true);
    setManualCohorts([{ patients: filtered, name: "" }]);
  };

  const handleSetCohortName = () => {
    const updated = [...manualCohorts];
    updated[0].name = cohortName || "Manual Cohort";
    setManualCohorts(updated);
    setShowCohortNameInput(false);
    setCohortName("");
  };

  const handleAnalyzeCohort = (cohort) => {
    navigate("/cohort-analysis", {
      state: { cohortData: cohort.patients, cohortName: cohort.name },
    });
  };
   const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("doctorEmail");
    if (onLogout) onLogout(); // update app state
    navigate("/login");
  };
  return (
    <div style={{ fontFamily: "Arial, sans-serif" }}>
      {/* Navbar */}
      <nav
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: "#004aad",
          padding: "10px 20px",
          color: "white",
          borderRadius: "8px",
        }}
      >
        <h2 style={{ margin: 0, cursor: "pointer" }} onClick={() => navigate("/")}>
          AI-Powered Cohort Analysis Platform
        </h2>
        <div style={{ display: "flex", gap: "20px" }}>
          <button style={navLinkStyle} onClick={() => navigate("/")}>
            Home
          </button>
          <button style={navLinkStyle} onClick={() => navigate("/patients")}>
            Patient Details
          </button>
          <button style={navLinkStyle} onClick={() => navigate("/cohort-segmentation")}>
            Cohort Analysis
          </button>
            <button style={navLinkStyle} onClick={() => navigate("/predictive-analysis")}>Predictive Analysis</button>
               <button style={logoutButtonStyle} onClick={handleLogout}>Logout</button>
        </div>
      </nav>

      {/* Description */}
      <div
        style={{
          background: "#f0f4ff",
          padding: "20px",
          borderRadius: "12px",
          margin: "20px",
          boxShadow: "0 3px 8px rgba(0,0,0,0.1)",
        }}
      >
        <p>
          Create patient cohorts for deeper medical insights. Choose between
          <strong> Manual Cohort</strong> (filter by specific conditions) or{" "}
          <strong>AI Cohort</strong> (automatically clustered based on health data).
          After creation, analyze cohort patterns, treatment outcomes, and risk factors.
        </p>
        <div style={{ marginTop: "15px" }}>
          <button
            style={featureButtonStyle(selectedFeature === "manual")}
            onClick={() => setSelectedFeature("manual")}
          >
            Manual Cohort
          </button>
          <button
            style={featureButtonStyle(selectedFeature === "ai")}
            onClick={() => setSelectedFeature("ai")}
          >
            AI Cohort
          </button>
        </div>
      </div>

      {/* Manual Cohort Section */}
      {selectedFeature === "manual" && (
        <div style={featureContainerStyle}>
          <h3>Manual Cohort Creation</h3>
          <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginBottom: "15px" }}>
            <input
              type="number"
              placeholder="Minimum Age"
              value={ageFilter}
              onChange={(e) => setAgeFilter(e.target.value)}
              style={inputStyle}
            />
            <input
              type="text"
              placeholder="Condition/Disease"
              value={conditionFilter}
              onChange={(e) => setConditionFilter(e.target.value)}
              style={inputStyle}
            />
            <button onClick={handleCreateManualCohort} style={buttonStyle}>
              Create Cohort
            </button>
          </div>

          {showCohortNameInput && (
            <div style={{ marginBottom: "15px" }}>
              <input
                type="text"
                placeholder="Enter Cohort Name"
                value={cohortName}
                onChange={(e) => setCohortName(e.target.value)}
                style={inputStyle}
              />
              <button onClick={handleSetCohortName} style={buttonStyle}>
                Save Cohort
              </button>
            </div>
          )}

          {/* Show Manual Cohort Card */}
          {manualCohorts.length > 0 && manualCohorts[0].patients.length > 0 && (
            <div
              style={{
                display: "flex",
                gap: "15px",
                flexWrap: "wrap",
                marginTop: "15px",
              }}
            >
              <div
                style={cardStyle}
                onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.03)")}
                onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}>
                <h4>{manualCohorts[0].name || "Manual Cohort"}</h4>
                <p>{manualCohorts[0].patients.length} patients</p>
                <ul style={{ maxHeight: "150px", overflowY: "auto" }}>
                  {manualCohorts[0].patients.map((p) => (
                    <li key={p.id}>{p.name} - Age {p.age} - {p.condition}</li>
                  ))}
                </ul>
                <button
                  style={buttonStyle}
                  onClick={() => handleAnalyzeCohort(manualCohorts[0])}
                >
                  Cohort Analysis
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* AI Cohort Section */}
      {selectedFeature === "ai" && (
        <div style={featureContainerStyle}>
          <h3>AI-Generated Cohorts</h3>
          <div style={{ display: "flex", gap: "15px", flexWrap: "wrap" }}>
            {aiCohorts.map((cohort, index) => (
              <div
                key={index}
                style={cardStyle}
                onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.03)")}
                onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}>
                <h4>{cohort.name}</h4>
                <p>{cohort.patients.length} patients</p>
                <ul style={{ maxHeight: "150px", overflowY: "auto" }}>
                  {cohort.patients.map((p) => (
                    <li key={p.id}>{p.name} - Age {p.age} - {p.condition}</li>
                  ))}
                </ul>
                <button
                  style={buttonStyle}
                  onClick={() => handleAnalyzeCohort(cohort)}
                >
                  Cohort Analysis
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Styles
const navLinkStyle = {
  padding: "8px 14px",
  borderRadius: "6px",
  border: "none",
  background: "#ddd",
  cursor: "pointer",
  fontWeight: "bold",
};

const featureButtonStyle = (active) => ({
  padding: "8px 20px",
  marginRight: "10px",
  borderRadius: "8px",
  border: "none",
  background: active ? "#004aad" : "#ccc",
  color: active ? "white" : "#333",
  cursor: "pointer",
  fontWeight: "bold",
});

const featureContainerStyle = {
  background: "#fff",
  padding: "20px",
  borderRadius: "12px",
  boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
  margin: "20px",
};

const inputStyle = {
  padding: "8px 12px",
  borderRadius: "6px",
  border: "1px solid #ccc",
};

const buttonStyle = {
  padding: "10px 18px",
  borderRadius: "8px",
  border: "none",
  background: "#004aad",
  color: "white",
  fontWeight: "bold",
  cursor: "pointer",
};

const cardStyle = {
  background: "#e0f0ff",
  padding: "15px",
  borderRadius: "10px",
  width: "300px",
  boxShadow: "0 2px 6px rgba(0,0,0,0.15)",
  transition: "transform 0.2s",
};
const logoutButtonStyle = {
  padding: "8px 14px",
  borderRadius: "6px",
  border: "none",
  background: "#ff4d4f",
  color: "white",
  cursor: "pointer",
  fontWeight: "bold",
  transition: "all 0.3s ease",
};
