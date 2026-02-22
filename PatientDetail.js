import React, { useState, useContext, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { PatientContext } from "../context/PatientContext";

const CollapsibleSection = ({ title, children }) => {
  const [open, setOpen] = useState(true);
  return (
    <div style={{
      border: "1px solid #cce0ff",
      borderRadius: "8px",
      marginBottom: "12px",
      background: "#f9fbff",
      boxShadow: "0 2px 4px rgba(0,0,0,0.05)"
    }}>
      <div
        style={{
          fontWeight: "600",
          cursor: "pointer",
          padding: "10px 16px",
          backgroundColor: "#e6f0ff",
          borderRadius: "8px 8px 0 0",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center"
        }}
        onClick={() => setOpen(!open)}
      >
        {title} <span>{open ? "▲" : "▼"}</span>
      </div>
      {open && <div style={{ padding: "10px 16px", color: "#333" }}>{children}</div>}
    </div>
  );
};

const PatientDetail = ({ onLogout }) => {
  const { id } = useParams(); // patient index
  const navigate = useNavigate();
  const { allPatients } = useContext(PatientContext); // ✅ get patients from context

  const [patient, setPatient] = useState(null);

  useEffect(() => {
    if (allPatients.length > 0) {
      const p = allPatients[parseInt(id)];
      setPatient(p || null);
    }
  }, [allPatients, id]);

  const age = patient?.birthdate
    ? Math.floor((new Date() - new Date(patient.birthdate)) / (365.25 * 24 * 60 * 60 * 1000))
    : "-";

  const ongoingCondition = patient?.current_condition || null;

  const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("doctorEmail");
    if (onLogout) onLogout();
    navigate("/login");
  };

  if (!allPatients.length) return <p style={{ padding: 20 }}>❌ No patients uploaded yet. Go back to Dashboard to upload CSV folder.</p>;
  if (!patient) return <p style={{ padding: 20 }}>❌ Patient not found.</p>;

  return (
    <div style={{ fontFamily: "Arial, sans-serif", minHeight: "100vh", background: "#eef2f7" }}>
      <nav style={navStyle}>
        <h2 style={{ margin: 0, cursor: "pointer" }} onClick={() => navigate("/")}>
          AI-Powered Cohort Analysis
        </h2>
        <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
          <button style={navLinkStyle} onClick={() => navigate("/")}>Home</button>
          <button style={navLinkStyle} onClick={() => navigate("/patients")}>Patient Details</button>
          <button style={navLinkStyle} onClick={() => navigate("/cohort-segmentation")}>Cohort Analysis</button>
          <button style={navLinkStyle} onClick={() => navigate("/predictive-analysis")}>Predictive Analysis</button>
          <button style={logoutButtonStyle} onClick={handleLogout}>Logout</button>
        </div>
      </nav>

      <div style={containerStyle}>
        <button onClick={() => navigate("/patients")} style={buttonStyle}>← Back to Patients List</button>

        <div style={patientCardStyle}>
          <h2 style={{ margin: "0 0 8px 0" }}>{patient.name || "Unnamed Patient"}</h2>
          <p style={{ margin: 0, color: "#555" }}>Age: {age} | Gender: {patient.gender || "N/A"}</p>
        </div>

        {ongoingCondition ? (
          <div>
            <div style={currentConditionStyle}>
              <h3 style={{ margin: "0 0 5px 0" }}>{ongoingCondition.DESCRIPTION || "N/A"}</h3>
              <p style={{ margin: 0, color: "#555" }}>
                Start: {ongoingCondition.START || "N/A"} | Stop: {ongoingCondition.STOP || "Ongoing"}
              </p>
            </div>

            <CollapsibleSection title="Medications">
              {patient.medications?.length
                ? patient.medications.map((m, idx) => (
                    <p key={idx}>{m.DESCRIPTION || "N/A"} ({m.START || "?"} - {m.STOP || "Ongoing"})</p>
                  ))
                : <p>No ongoing medications.</p>}
            </CollapsibleSection>

            <CollapsibleSection title="Observations">
              {patient.observations?.length
                ? patient.observations.map((o, idx) => (
                    <p key={idx}>{o.DESCRIPTION || "N/A"} — {o.VALUE || "N/A"} {o.UNITS || ""}</p>
                  ))
                : <p>No observations.</p>}
            </CollapsibleSection>

            <CollapsibleSection title="Procedures">
              {patient.procedures?.length
                ? patient.procedures.map((p, idx) => <p key={idx}>{p.DESCRIPTION || "N/A"}</p>)
                : <p>No ongoing procedures.</p>}
            </CollapsibleSection>
          </div>
        ) : <p style={{ color: "#777" }}>No current ongoing condition for this patient.</p>}
      </div>
    </div>
  );
};

// ------------------------
// Styles
// ------------------------
const containerStyle = {
  maxWidth: 900,
  margin: "20px auto",
  padding: 20,
  background: "#fff",
  borderRadius: 10,
  boxShadow: "0 3px 8px rgba(0,0,0,0.1)"
};

const patientCardStyle = {
  background: "#e6f0ff",
  padding: "15px 20px",
  borderRadius: 10,
  marginBottom: 20
};

const currentConditionStyle = {
  background: "#cce0ff",
  padding: "12px 18px",
  borderRadius: 8,
  marginBottom: 15
};

const navStyle = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  background: "#004aad",
  padding: "10px 20px",
  color: "white",
  borderRadius: 8,
};

const navLinkStyle = {
  padding: "8px 14px",
  borderRadius: 6,
  border: "none",
  background: "#ddd",
  cursor: "pointer",
  fontWeight: "bold",
  transition: "all 0.3s ease",
};

const buttonStyle = {
  marginBottom: 15,
  padding: "8px 16px",
  borderRadius: 6,
  border: "none",
  background: "#004aad",
  color: "white",
  cursor: "pointer",
  fontWeight: "bold"
};

const logoutButtonStyle = {
  padding: "8px 14px",
  borderRadius: 6,
  border: "none",
  background: "#ff4d4f",
  color: "white",
  cursor: "pointer",
  fontWeight: "bold",
  transition: "all 0.3s ease",
};

export default PatientDetail;