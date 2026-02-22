import React, { useState, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar
} from "recharts";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

const COLORS = ["#004aad", "#00C49F", "#FFBB28", "#FF8042"];

export default function CohortAnalysis({ onLogout }) {
  const navigate = useNavigate();
  const location = useLocation();
  const cohort = location.state?.cohortData || [];
  const cohortName = location.state?.cohortName || "Unnamed Cohort";
  const [hoveredRow, setHoveredRow] = useState(null);

  const pageRef = useRef(); // Reference for the whole page

  if (!cohort.length)
    return <p style={{ textAlign: "center", padding: "20px" }}>No cohort data available.</p>;

  // --- Data Analysis ---
  const avgAge = (cohort.reduce((sum, p) => sum + p.age, 0) / cohort.length).toFixed(1);
  const genderCount = { Male: cohort.filter((p) => p.gender === "Male").length, Female: cohort.filter((p) => p.gender === "Female").length };
  const commonConditions = [...new Set(cohort.map((p) => p.condition))];

  const diseaseProgression = [
    { month: "Jan", Diabetes: 80, Hypertension: 70, Asthma: 60, "Heart Disease": 90 },
    { month: "Feb", Diabetes: 75, Hypertension: 65, Asthma: 55, "Heart Disease": 85 },
    { month: "Mar", Diabetes: 68, Hypertension: 58, Asthma: 50, "Heart Disease": 78 },
    { month: "Apr", Diabetes: 60, Hypertension: 50, Asthma: 48, "Heart Disease": 70 },
  ];

  const treatmentPerformance = [
    { treatment: "Metformin", successRate: 85, relapseRate: 10 },
    { treatment: "Insulin", successRate: 78, relapseRate: 15 },
    { treatment: "Amlodipine", successRate: 82, relapseRate: 12 },
    { treatment: "Losartan", successRate: 88, relapseRate: 8 },
    { treatment: "Aspirin", successRate: 75, relapseRate: 20 },
  ];

  const riskFactors = [
    { factor: "Age > 50", riskScore: "High", note: "Older patients had slower recovery." },
    { factor: "High BP (>140/90)", riskScore: "Medium", note: "Linked to heart disease relapse." },
    { factor: "Heart Rate > 85 bpm", riskScore: "Medium", note: "Indicates stress or complications." },
  ];

  // --- Download PDF Function ---
  const downloadPDF = () => {
    const input = pageRef.current;
    html2canvas(input, { scale: 2 }).then((canvas) => {
      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
      pdf.save(`${cohortName}_CohortAnalysis.pdf`);
    });
  };

  const navLinkStyle = {
    padding: "8px 14px",
    borderRadius: "6px",
    border: "none",
    background: "#ddd",
    cursor: "pointer",
    fontWeight: "bold",
  };
   const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("doctorEmail");
    if (onLogout) onLogout(); // update app state
    navigate("/login");
  };
  const styles = {
    container: { minHeight: "100vh", backgroundColor: "#f0f4f8", padding: "30px 2rem 2rem 2rem", fontFamily: "Arial, sans-serif" },
    title: { textAlign: "center", fontSize: "2.2rem", fontWeight: "bold", color: "#0f172a", textShadow: "1px 1px 3px rgba(0,0,0,0.1)" },
    summaryCards: { display: "flex", flexWrap: "wrap", gap: "1rem", marginBottom: "2rem", justifyContent: "center" },
    summaryCard: { flex: "1 1 200px", padding: "1.5rem", borderRadius: "16px", boxShadow: "0 4px 12px rgba(0,0,0,0.1)", textAlign: "center", color: "white", fontWeight: "500" },
    avgAgeCard: { backgroundColor: "#4ade80" },
    totalPatientsCard: { backgroundColor: "#60a5fa" },
    genderCard: { backgroundColor: "#facc15", color: "#1e293b" },
    conditionsCard: { backgroundColor: "#f472b6" },
    tableContainer: { marginBottom: "2rem", backgroundColor: "white", padding: "1rem", borderRadius: "16px", boxShadow: "0 2px 12px rgba(0,0,0,0.1)", overflowX: "auto" },
    table: { width: "100%", borderCollapse: "collapse" },
    th: { background: "linear-gradient(90deg, #dbeafe, #bfdbfe)", padding: "0.5rem", border: "1px solid #e2e8f0" },
    td: { padding: "0.5rem", border: "1px solid #e2e8f0", transition: "0.3s" },
    tdHover: { backgroundColor: "#f1f5f9" },
    chartContainer: { marginBottom: "2rem", backgroundColor: "white", padding: "1.5rem", borderRadius: "16px", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" },
    chartTitle: { fontWeight: "bold", fontSize: "1.2rem", color: "#1e3a8a", marginBottom: "1rem", textAlign: "center" },
    chartNote: { textAlign: "center", marginTop: "0.5rem", color: "#4b5563", fontStyle: "italic" },
    riskHigh: { color: "red", fontWeight: "bold" },
    riskMedium: { color: "orange", fontWeight: "bold" },
    riskLow: { color: "green", fontWeight: "bold" },
  };

  return (
    <>
      {/* FIXED NAVBAR */}
      <nav style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "#004aad", padding: "10px 20px", color: "white", borderRadius: "8px" }}>
        <h2 style={{ margin: 0, cursor: "pointer" }} onClick={() => navigate("/")}>
          AI-Powered Cohort Analysis Platform
        </h2>
        <div style={{ display: "flex", gap: "20px" }}>
          <button style={navLinkStyle} onClick={() => navigate("/")}>Home</button>
          <button style={navLinkStyle} onClick={() => navigate("/patients")}>Patient Details</button>
          <button style={navLinkStyle} onClick={() => navigate("/cohort-segmentation")}>Cohort Analysis</button>
          <button style={navLinkStyle} onClick={() => navigate("/predictive-analysis")}>Predictive Analysis</button>
             <button style={logoutButtonStyle} onClick={handleLogout}>Logout</button>
        </div>
      </nav>

      <div ref={pageRef} style={styles.container}>
        <h1 style={styles.title}>{cohortName} – Cohort Analysis Dashboard</h1>

        {/* DOWNLOAD BUTTON */}
        <button
          onClick={downloadPDF}
          style={{ margin: "0 auto 2rem auto", display: "block", padding: "10px 20px", borderRadius: "8px", backgroundColor: "#4caf50", color: "white", border: "none", fontWeight: "bold", cursor: "pointer" }}
        >
          Download PDF Report
        </button>

        {/* SUMMARY CARDS */}
        <div style={styles.summaryCards}>
          <div style={{ ...styles.summaryCard, ...styles.avgAgeCard }}>
            <h3>Average Age</h3>
            <p>{avgAge} yrs</p>
          </div>
          <div style={{ ...styles.summaryCard, ...styles.totalPatientsCard }}>
            <h3>Total Patients</h3>
            <p>{cohort.length}</p>
          </div>
          <div style={{ ...styles.summaryCard, ...styles.genderCard }}>
            <h3>Gender Ratio</h3>
            <p>{genderCount.Male} : {genderCount.Female}</p>
          </div>
          <div style={{ ...styles.summaryCard, ...styles.conditionsCard }}>
            <h3>Common Conditions</h3>
            <p>{commonConditions.join(", ")}</p>
          </div>
        </div>

        {/* PATIENT TABLE */}
        <div style={styles.tableContainer}>
          <h3>Patient Summary</h3>
          <table style={styles.table}>
            <thead>
              <tr>
                {["Name", "Age", "Gender", "Condition", "Blood Pressure", "Heart Rate", "Medications"].map((header) => (
                  <th key={header} style={styles.th}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cohort.map((p, idx) => (
                <tr key={idx} onMouseEnter={() => setHoveredRow(idx)} onMouseLeave={() => setHoveredRow(null)} style={hoveredRow === idx ? styles.tdHover : {}}>
                  <td style={styles.td}>{p.name}</td>
                  <td style={styles.td}>{p.age}</td>
                  <td style={styles.td}>{p.gender}</td>
                  <td style={styles.td}>{p.condition}</td>
                  <td style={styles.td}>{p.bloodPressure}</td>
                  <td style={styles.td}>{p.heartRate}</td>
                  <td style={styles.td}>{p.medications.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* DISEASE PROGRESSION */}
        <div style={styles.chartContainer}>
          <h3 style={styles.chartTitle}>Disease Progression Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={diseaseProgression}>
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              {Object.keys(diseaseProgression[0]).filter((key) => key !== "month").map((disease, i) => (
                <Line key={disease} type="monotone" dataKey={disease} stroke={COLORS[i % COLORS.length]} strokeWidth={2} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* TREATMENT OUTCOME */}
        <div style={styles.chartContainer}>
          <h3 style={styles.chartTitle}>Treatment Success & Relapse Analysis</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={treatmentPerformance}>
              <XAxis dataKey="treatment" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="successRate" fill="#00C49F" name="Success Rate (%)" />
              <Bar dataKey="relapseRate" fill="#FF8042" name="Relapse Rate (%)" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* RISK FACTORS */}
        <div style={styles.tableContainer}>
          <h3>Risk Factor Analysis</h3>
          <table style={styles.table}>
            <thead>
              <tr>
                {["Risk Factor", "Risk Level", "Observation"].map((header) => (
                  <th key={header} style={styles.th}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {riskFactors.map((r, idx) => (
                <tr key={idx}>
                  <td style={styles.td}>{r.factor}</td>
                  <td style={{
                    ...styles.td,
                    ...(r.riskScore === "High" ? styles.riskHigh : r.riskScore === "Medium" ? styles.riskMedium : styles.riskLow)
                  }}>{r.riskScore}</td>
                  <td style={styles.td}>{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

      </div>
    </>
  );
}
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
