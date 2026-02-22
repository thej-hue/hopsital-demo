import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

export default function PatientPredictiveDetail({ doctorEmail, onLogout }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predictedRisk, setPredictedRisk] = useState(null);
  const [predictedFutureRisk, setPredictedFutureRisk] = useState(null);
  const [recommendedTreatment, setRecommendedTreatment] = useState(null);
  const [actualScores, setActualScores] = useState({ current: null, future: null });

  const navLinkStyle = {
    background: "#dad4ebff",
    color: "#202022ff",
    border: "none",
    padding: "8px 16px",
    borderRadius: "5px",
    fontWeight: "bold",
    cursor: "pointer",
    marginLeft: "10px",
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

  const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("doctorEmail");
    if (onLogout) onLogout();
    navigate("/login");
  };

  const Navbar = () => (
    <nav
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "10px 20px",
        background: "#1e40af",
        color: "white",
      }}
    >
      <h2 style={{ margin: 0, cursor: "pointer" }} onClick={() => navigate("/")}>
        AI-Powered Cohort Analysis
      </h2>
      <div>
        <button style={navLinkStyle} onClick={() => navigate("/")}>
          Home
        </button>
        <button
          style={navLinkStyle}
          onClick={() =>
            navigate(`/patients?email=${encodeURIComponent(doctorEmail)}`)
          }
        >
          Patient Details
        </button>
        <button
          style={navLinkStyle}
          onClick={() => navigate("/cohort-segmentation")}
        >
          Cohort Analysis
        </button>
        <button
          style={navLinkStyle}
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

  useEffect(() => {
    const fetchPatientData = async () => {
      try {
        const res = await axios.get("http://127.0.0.1:5000/predictive_analysis", {
          params: { email: doctorEmail },
        });

        const data = Array.isArray(res.data)
          ? res.data
          : res.data?.patients || res.data?.data || [];

        const p = data.find((pt) => String(pt.id) === String(id));
        setPatient(p || null);
        if (!p) return;

        const payload = {
          AGE: p.age || 0,
          procedure_severity: 1,
          medication_intensity: 1,
          treatment_response: 0,
          observation_value: 0,
        };

        // ✅ Current Risk
        const riskRes = await axios.post("http://127.0.0.1:5000/predict_risk", payload);
        const currentRisk = riskRes.data?.predicted_risk || "N/A";
        const currentScore = riskRes.data?.confidence_score || null;

        // ✅ Future Risk
        const futureRes = await axios.post("http://127.0.0.1:5000/predict_future_risk", payload);
        const futureRisk = futureRes.data?.predicted_future_risk || "N/A";
        const futureScore = futureRes.data?.predicted_future_risk || null;
        console.log("🧾 Future risk API response:", futureRes.data);

        setPredictedRisk(currentRisk);
        setPredictedFutureRisk(futureRisk);
        setActualScores({ current: currentScore, future: futureScore });

        // ✅ Treatment Recommendation
        const treatRes = await axios.post("http://127.0.0.1:5000/recommend_treatment", {
          condition_desc: p.condition || "Unknown condition",
          email: doctorEmail,
        });
        setRecommendedTreatment(
          treatRes.data?.recommended_treatment || "No recommendation available"
        );
      } catch (err) {
        console.error("❌ Error fetching patient data:", err);
        alert("Failed to fetch prediction details");
      } finally {
        setLoading(false);
      }
    };
    

    if (doctorEmail && id) fetchPatientData();
  }, [id, doctorEmail]);

  if (loading)
    return (
      <div>
        <Navbar />
        <p style={{ padding: "30px" }}>Loading patient details...</p>
      </div>
    );

  if (!patient)
    return (
      <div>
        <Navbar />
        <h2 style={{ color: "#1e40af", padding: "30px" }}>Patient not found</h2>
      </div>
    );

  const riskColor =
    predictedRisk?.toLowerCase().includes("high")
      ? "#dc2626"
      : predictedRisk?.toLowerCase().includes("medium")
      ? "#f59e0b"
      : "#16a34a";

  // ✅ Risk data for trend line (using actual numeric scores)
  const riskData = [
    { name: "Current", value: actualScores.current || 0 },
    { name: "Future (5yr)", value: actualScores.future || 0 },
  ];

  return (
    <>
      <Navbar />
      <div
        style={{
          fontFamily: "Arial",
          background: "#f0f4f8",
          minHeight: "100vh",
          padding: "30px",
        }}
      >
        <div
          style={{
            background: "white",
            padding: "25px",
            borderRadius: "12px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
            maxWidth: "1100px",
            margin: "auto",
          }}
        >
          <button
            onClick={() => navigate("/predictive-analysis")}
            style={{
              background: "#1e40af",
              color: "white",
              padding: "8px 14px",
              borderRadius: "5px",
              border: "none",
              marginBottom: "15px",
              cursor: "pointer",
            }}
          >
            ← Back
          </button>

          <h2 style={{ color: "#1e40af" }}>{patient.name}</h2>
          <p><strong>ID:</strong> {patient.id}</p>
          <p><strong>Age:</strong> {patient.age}</p>
          <p><strong>Gender:</strong> {patient.gender}</p>
          <p><strong>Condition:</strong> {patient.condition}</p>

          {/* ✅ Risk Trend */}
          <div style={{ marginTop: "30px" }}>
            <h3 style={{ color: "#1e40af" }}>Health Risk Trend</h3>
            <div style={{ marginTop: "10px", fontSize: "16px" }}>
              <p>
                <strong>Current Risk:</strong> 
                {actualScores.current !== null && (
                  <> {actualScores.current.toFixed(3)}</>
                )}
              </p>
              <p>
                <strong>Future Risk:</strong>
                {actualScores.future !== null && (
                  <> {actualScores.future.toFixed(3)}</>
                )}
              </p>
            </div>

            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={riskData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(value) => [`Actual Score: ${value.toFixed(3)}`, "Risk Level"]} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={riskColor}
                  strokeWidth={3}
                  dot={{ r: 6, fill: riskColor }}
                  activeDot={{ r: 9 }}
                  isAnimationActive={true}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* ✅ Treatment Recommendation */}
          <div
            style={{
              marginTop: "30px",
              background: "#ecfdf5",
              padding: "20px",
              borderRadius: "10px",
              borderLeft: "6px solid #16a34a",
            }}
          >
            <h3 style={{ color: "#16a34a" }}>Recommended Treatment</h3>
            <p>{recommendedTreatment}</p>
          </div>
        </div>
      </div>
    </>
  );
}
