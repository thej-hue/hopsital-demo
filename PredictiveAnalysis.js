import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

// 🔴 Logout button style
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

export default function PredictiveAnalysis({ doctorEmail, onLogout }) {
  const navigate = useNavigate();
  const [highRiskPatients, setHighRiskPatients] = useState([]);
  const [allPatients, setAllPatients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchId, setSearchId] = useState("");
  const [filterName, setFilterName] = useState("");
  const [filterAge, setFilterAge] = useState("");

  // ✅ Logout handler
  const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("doctorEmail");
    if (onLogout) onLogout();
    navigate("/login");
  };

  // ✅ Navbar
  const Navbar = () => (
    <nav
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "8px 20px",
        background: "#5867b3ff",
        color: "white",
      }}
    >
      <h2 style={{ margin: 0, cursor: "pointer" }} onClick={() => navigate("/")}>
        AI-Powered Cohort Analysis
      </h2>
      <div>
         <button style={navButtonStyle} onClick={() => navigate("/")}>
          Home
        </button>
        <button
          style={navButtonStyle}
          onClick={() => navigate(`/patients?email=${encodeURIComponent(doctorEmail)}`)}
        >
          Patient Details
        </button>
        <button
          style={navButtonStyle}
          onClick={() => navigate("/cohort-segmentation")}
        >
          Cohort Analysis
        </button>
        <button
          style={navButtonStyle}
          onClick={() => navigate("/predictive-analysis")}
        >
          Predictive Analysis
        </button>
        <button style={logoutButtonStyle} onClick={handleLogout}>
          Logout
        </button>
      </div>
    </nav>
  );

  const navButtonStyle = {
    background: "#d4d0d0ff",
    color: "#111111ff",
    border: "none",
    padding: "8px 16px",
    borderRadius: "5px",
    fontWeight: "bold",
    cursor: "pointer",
    marginLeft: "10px",
  };

  // ✅ Fetch data
  useEffect(() => {
    const fetchPatients = async () => {
      setLoading(true);
      try {
        const [highRiskRes, allRes] = await Promise.all([
          axios.get("http://127.0.0.1:5000/high_risk_patients", {
            params: { email: doctorEmail },
          }),
          axios.get("http://127.0.0.1:5000/predictive_analysis", {
            params: { email: doctorEmail },
          }),
        ]);

        // Store data
        setHighRiskPatients(highRiskRes.data || []);
        setAllPatients(allRes.data || []);

        console.log("✅ Data fetched successfully:", {
          highRisk: highRiskRes.data.length,
          all: allRes.data.length,
        });
      } catch (err) {
        console.error("❌ Error fetching patients:", err);
        alert("Failed to fetch patient data. Check Flask server or email.");
      } finally {
        setLoading(false);
      }
    };

    if (doctorEmail) fetchPatients();
  }, [doctorEmail]);

  // 🔍 Search by patient ID
  const handleSearchById = () => {
    if (!searchId.trim()) {
      alert("Please enter a patient ID to search.");
      return;
    }

    const found = allPatients.find(
      (p) => String(p.id).trim() === String(searchId).trim()
    );

    if (found) {
      console.log("🔍 Found patient:", found);
      navigate(`/patient-predictive/${found.id}`);
    } else {
      alert("❌ Patient not found in database.");
    }
  };

  // 🩺 Filter high-risk patients
  const filteredHighRisk = highRiskPatients
    .filter((p) =>
      filterName ? p.name?.toLowerCase().includes(filterName.toLowerCase()) : true
    )
    .filter((p) => (filterAge ? p.age >= parseInt(filterAge) : true));

  // ✅ Render UI
  return (
    <>
      <Navbar />
      <div
        style={{
          fontFamily: "Arial, sans-serif",
          background: "#f0f4f8",
          minHeight: "100vh",
          padding: "30px",
        }}
      >
        <h2 style={{ color: "#1e40af" }}>Predictive Analysis Dashboard</h2>

        {/* 🔍 Search by ID */}
        <div style={{ marginBottom: "20px" }}>
          <input
            type="text"
            placeholder="Enter Patient ID"
            value={searchId}
            onChange={(e) => setSearchId(e.target.value)}
            style={{
              padding: "8px",
              marginRight: "10px",
              borderRadius: "5px",
              border: "1px solid #ccc",
            }}
          />
          <button
            onClick={handleSearchById}
            style={{
              padding: "8px 12px",
              background: "#1e40af",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            Search
          </button>
        </div>

        {/* 🧩 Filters */}
        <div style={{ marginBottom: "20px" }}>
          <input
            type="text"
            placeholder="Filter by Name"
            value={filterName}
            onChange={(e) => setFilterName(e.target.value)}
            style={{
              padding: "8px",
              marginRight: "10px",
              borderRadius: "5px",
              border: "1px solid #ccc",
            }}
          />
          <input
            type="number"
            placeholder="Filter by Min Age"
            value={filterAge}
            onChange={(e) => setFilterAge(e.target.value)}
            style={{
              padding: "8px",
              borderRadius: "5px",
              border: "1px solid #ccc",
            }}
          />
        </div>

        {/* 📋 High-Risk Table */}
        {loading ? (
          <p>Loading patient data...</p>
        ) : filteredHighRisk.length === 0 ? (
          <p>No High-Risk patients found.</p>
        ) : (
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              background: "white",
              borderRadius: "10px",
              overflow: "hidden",
              boxShadow: "0 4px 10px rgba(0,0,0,0.1)",
            }}
          >
            <thead style={{ background: "#1e40af", color: "white" }}>
              <tr>
                <th style={{ padding: "10px" }}>Patient ID</th>
                <th style={{ padding: "10px" }}>Name</th>
                <th style={{ padding: "10px" }}>Age</th>
                <th style={{ padding: "10px" }}>Condition</th>
                <th style={{ padding: "10px" }}>Predicted Risk</th>
              </tr>
            </thead>
            <tbody>
              {filteredHighRisk.map((p) => (
                <tr
                  key={p.id}
                  onClick={() => navigate(`/patient-predictive/${p.id}`)}
                  style={{
                    borderBottom: "1px solid #ddd",
                    cursor: "pointer",
                  }}
                >
                  <td style={{ padding: "8px" }}>{p.id}</td>
                  <td style={{ padding: "8px" }}>{p.name}</td>
                  <td style={{ padding: "8px" }}>{p.age}</td>
                  <td style={{ padding: "8px" }}>{p.condition}</td>
                  <td
                    style={{
                      padding: "8px",
                      fontWeight: "bold",
                      color:
                        p.predicted_risk === "High"
                          ? "#dc2626"
                          : p.predicted_risk === "Medium"
                          ? "#f59e0b"
                          : "#16a34a",
                    }}
                  >
                    {p.predicted_risk}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
