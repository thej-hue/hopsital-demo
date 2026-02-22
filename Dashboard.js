import React, { useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import Papa from "papaparse";
import { PatientContext } from "../context/PatientContext";
import doctorImage from "../assets/doctor_cartoon.png";

export default function Dashboard({ onLogout }) {
  const { updatePatients } = useContext(PatientContext);

  const [message, setMessage] = useState("");
  const [uploading, setUploading] = useState(false);
  const [highRiskPatients, setHighRiskPatients] = useState([]);

  const navigate = useNavigate();

  const doctorEmail = localStorage.getItem("doctorEmail");
  const doctorName = doctorEmail ? doctorEmail.split("@")[0] : "Doctor";

  // ===============================
  // FOLDER UPLOAD HANDLER (FINAL)
  // ===============================
  const handleFilesUpload = (event) => {
    const filesList = Array.from(event.target.files).filter((file) =>
      file.name.toLowerCase().endsWith(".csv")
    );

    if (!filesList.length) {
      setMessage("❌ No CSV files found in the selected folder.");
      return;
    }

    setUploading(true);
    let allPatientsTemp = [];
    let processedFiles = 0;

    filesList.forEach((file) => {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          // 🔹 Normalize Synthea / medical CSV structure
          const normalizedPatients = results.data.map((row) => ({
            id: row.Id || row.ID || row.id || "",
            name: row.NAME || row.name || "Unknown",
            age: row.AGE || row.age || "N/A",
            condition:
              row.CONDITION ||
              row.condition ||
              row.DIAGNOSIS ||
              "N/A",
            risk: (row.RISK || row.risk || "low").toLowerCase(),
          }));

          allPatientsTemp = [...allPatientsTemp, ...normalizedPatients];
          processedFiles++;

          if (processedFiles === filesList.length) {
            // ✅ Save globally
            updatePatients(allPatientsTemp);

            // ✅ High-risk filter works now
            const highRisk = allPatientsTemp.filter(
              (p) => p.risk === "high"
            );

            setHighRiskPatients(highRisk);
            setUploading(false);
            setMessage(
              `✅ ${allPatientsTemp.length} records loaded (${highRisk.length} high-risk)`
            );
            setTimeout(() => setMessage(""), 5000);
          }
        },
        error: () => {
          setUploading(false);
          setMessage("❌ Error parsing one or more CSV files.");
          setTimeout(() => setMessage(""), 5000);
        },
      });
    });
  };

  const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("doctorEmail");
    if (onLogout) onLogout();
    navigate("/login");
  };

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "Poppins, sans-serif" }}>
      {/* LEFT PANEL */}
      <div
        style={{
          width: "32%",
          background: "#2bb3b1",
          color: "white",
          padding: "60px 40px",
          position: "relative",
        }}
      >
        <h1 style={{ fontSize: "48px", marginBottom: "40px" }}>
          Hey Dr. {doctorName}!
        </h1>

        <div style={badgeStyle}>✔ Organized Layers</div>
        <div style={badgeStyle}>✔ Cohort Segmentation</div>
        <div style={badgeStyle}>✔ Predictive Analytics</div>
        <div style={badgeStyle}>✔ AI Healthcare Insights</div>

        <img
          src={doctorImage}
          alt="Doctor"
          style={{
            position: "absolute",
            bottom: 0,
            right: -40,
            height: "350px",
          }}
        />
      </div>

      {/* RIGHT PANEL */}
      <div style={{ width: "68%", display: "flex", background: "#f4f6f9" }}>
        {/* Sidebar */}
        <div style={sidebarStyle}>
          <div style={menuItemStyle} onClick={() => navigate("/")}>🏠</div>
          <div style={menuItemStyle} onClick={() => navigate("/patients")}>👥</div>
          <div style={menuItemStyle} onClick={() => navigate("/cohort-segmentation")}>📊</div>
          <div style={menuItemStyle} onClick={() => navigate("/predictive-analysis")}>🤖</div>

          <div style={{ marginTop: "auto", marginBottom: "30px" }}>
            <button style={logoutCircleStyle} onClick={handleLogout}>⎋</button>
          </div>
        </div>

        {/* Content */}
        <div style={{ flex: 1, padding: "40px" }}>
          <h2>AI-driven Stratified Health Analytics for Precision Care</h2>
          <p style={{ color: "#666", marginBottom: "30px" }}>
            Welcome to your dashboard
          </p>

          <div style={uploadCardStyle}>
            <h3 style={{ marginBottom: "20px" }}>Upload Medical Records Folder</h3>

            <input
              type="file"
              multiple
              webkitdirectory="true"
              directory=""
              accept=".csv"
              onChange={handleFilesUpload}
              style={{ display: "none" }}
              id="csv-folder-upload"
            />

            <label
              htmlFor="csv-folder-upload"
              style={{
                ...uploadModernButton,
                opacity: uploading ? 0.6 : 1,
              }}
            >
              {uploading ? "Uploading..." : "📁 Upload CSV Folder"}
            </label>

            {message && (
              <p
                style={{
                  marginTop: "20px",
                  color: message.startsWith("❌") ? "red" : "green",
                  fontWeight: "bold",
                }}
              >
                {message}
              </p>
            )}
          </div>

          {highRiskPatients.length > 0 && (
            <div>
              <h3 style={{ marginBottom: "15px" }}>High-Risk Patients</h3>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "20px" }}>
                {highRiskPatients.map((p, idx) => (
                  <div key={idx} style={patientCardStyle}>
                    <strong>{p.name}</strong>
                    <p>Age: {p.age}</p>
                    <p>Condition: {p.condition}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// =======================
// Styles
// =======================
const badgeStyle = {
  background: "white",
  color: "black",
  padding: "12px 20px",
  borderRadius: "30px",
  marginBottom: "15px",
  width: "fit-content",
};

const menuItemStyle = {
  fontSize: "26px",
  marginBottom: "40px",
  cursor: "pointer",
};

const logoutCircleStyle = {
  border: "none",
  background: "#ff4d4f",
  color: "white",
  borderRadius: "50%",
  width: "40px",
  height: "40px",
  cursor: "pointer",
};

const uploadModernButton = {
  padding: "15px 35px",
  borderRadius: "12px",
  border: "none",
  background: "#2bb3b1",
  color: "white",
  fontSize: "16px",
  fontWeight: "bold",
  cursor: "pointer",
};

const sidebarStyle = {
  width: "80px",
  background: "white",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  paddingTop: "40px",
  boxShadow: "2px 0 10px rgba(0,0,0,0.05)",
};

const uploadCardStyle = {
  background: "white",
  padding: "40px",
  borderRadius: "15px",
  boxShadow: "0 6px 20px rgba(0,0,0,0.08)",
  textAlign: "center",
  marginBottom: "30px",
};

const patientCardStyle = {
  background: "white",
  padding: "20px",
  borderRadius: "12px",
  boxShadow: "0 4px 15px rgba(0,0,0,0.06)",
  minWidth: "180px",
};