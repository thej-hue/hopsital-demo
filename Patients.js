import React, { useContext, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PatientContext } from "../context/PatientContext";

export default function Patients({ onLogout }) {
  const navigate = useNavigate();
  const { allPatients } = useContext(PatientContext);

  const [filters, setFilters] = useState({
    id: "",
    name: "",
    minAge: "",
    gender: "All",
    condition: "",
  });

  const email = localStorage.getItem("doctorEmail");

  const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("doctorEmail");
    if (onLogout) onLogout();
    navigate("/login");
  };

  const handleFilterChange = (e) => {
    setFilters({ ...filters, [e.target.name]: e.target.value });
  };

  const filteredPatients = allPatients.filter((p) => {
    const matchesId = !filters.id || String(p.id).includes(filters.id);
    const matchesName =
      !filters.name ||
      p.name.toLowerCase().includes(filters.name.toLowerCase());
    const matchesAge =
      !filters.minAge || (p.age && Number(p.age) >= Number(filters.minAge));
    const matchesGender =
      filters.gender === "All" ||
      (p.gender &&
        p.gender.toLowerCase() === filters.gender.toLowerCase());
    const matchesCondition =
      !filters.condition ||
      (p.condition &&
        p.condition.toLowerCase().includes(filters.condition.toLowerCase()));

    return (
      matchesId &&
      matchesName &&
      matchesAge &&
      matchesGender &&
      matchesCondition
    );
  });

  return (
    <div style={{ fontFamily: "Arial, sans-serif", minHeight: "100vh", background: "#f5f7fb" }}>
      {/* NAVBAR */}
      <nav style={navStyle}>
        <h2 style={{ margin: 0, cursor: "pointer" }} onClick={() => navigate("/")}>
          AI-Powered Cohort Analysis
        </h2>
        <div style={{ display: "flex", gap: "15px" }}>
          <button style={navLinkStyle} onClick={() => navigate("/")}>Home</button>
          <button style={navLinkStyle} onClick={() => navigate("/patients")}>Patient Details</button>
          <button style={navLinkStyle} onClick={() => navigate("/cohort-segmentation")}>Cohort Analysis</button>
          <button style={navLinkStyle} onClick={() => navigate("/predictive-analysis")}>Predictive Analysis</button>
          <button style={logoutButtonStyle} onClick={handleLogout}>Logout</button>
        </div>
      </nav>

      <div style={{ padding: "30px" }}>
        <h1 style={{ color: "#004aad", marginBottom: "20px" }}>Patient Details</h1>

        {/* FILTERS */}
        <div style={filterContainerStyle}>
          <input name="id" placeholder="Filter by ID" value={filters.id} onChange={handleFilterChange} style={filterInputStyle} />
          <input name="name" placeholder="Filter by Name" value={filters.name} onChange={handleFilterChange} style={filterInputStyle} />
          <input name="minAge" type="number" placeholder="Age ≥" value={filters.minAge} onChange={handleFilterChange} style={filterInputStyle} />
          <select name="gender" value={filters.gender} onChange={handleFilterChange} style={filterInputStyle}>
            <option value="All">All Genders</option>
            <option value="m">Male</option>
            <option value="f">Female</option>
          </select>
          <input name="condition" placeholder="Filter by Condition" value={filters.condition} onChange={handleFilterChange} style={filterInputStyle} />
        </div>

        {allPatients.length === 0 ? (
          <p style={{ fontSize: "16px", color: "red", fontWeight: "bold" }}>
            ❌ No data loaded. Please upload the CSV folder from Dashboard first.
          </p>
        ) : filteredPatients.length === 0 ? (
          <p style={{ fontSize: "16px", color: "#555" }}>
            No patient data found with applied filters.
          </p>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "20px" }}>
            <thead>
              <tr style={{ background: "#004aad", color: "white" }}>
                <th style={thStyle}>ID</th>
                <th style={thStyle}>Name</th>
                <th style={thStyle}>Age</th>
                <th style={thStyle}>Gender</th>
                <th style={thStyle}>Condition</th>
              </tr>
            </thead>
            <tbody>
              {filteredPatients.map((p, index) => (
                <tr key={index} style={{ background: index % 2 === 0 ? "#fff" : "#f0f4ff" }}>
                  <td style={tdStyle}>{p.id}</td>
                  <td style={tdStyle}>{p.name}</td>
                  <td style={tdStyle}>{p.age || "N/A"}</td>
                  <td style={tdStyle}>{p.gender || "N/A"}</td>
                  <td style={tdStyle}>{p.condition || "N/A"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ---- STYLES ----
const navStyle = { display: "flex", justifyContent: "space-between", alignItems: "center", background: "#004aad", padding: "10px 20px", color: "white", borderRadius: "8px" };
const thStyle = { padding: "12px", textAlign: "left" };
const tdStyle = { padding: "12px" };
const filterContainerStyle = { display: "flex", gap: "12px", marginBottom: "25px", flexWrap: "wrap", background: "#f0f4ff", padding: "20px", borderRadius: "12px" };
const filterInputStyle = { padding: "10px 14px", borderRadius: "10px", border: "1px solid #c0c4cc", fontSize: "14px" };
const navLinkStyle = { padding: "8px 14px", borderRadius: "6px", border: "none", background: "#e6eaee", cursor: "pointer", fontWeight: "600" };
const logoutButtonStyle = { padding: "8px 14px", borderRadius: "6px", border: "none", background: "#ff4d4f", color: "white", cursor: "pointer", fontWeight: "bold" };